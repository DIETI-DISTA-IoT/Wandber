from confluent_kafka import Consumer, KafkaError
import time
import json
import threading
import numpy as np
import wandb
import plotly.express as px
from OpenFAIR import EventType
import string
import random

# Patterns to match different types of Kafka topics
topics_dict = {
    # "anomalies": "^.*_anomalies$",  # Topics containing anomalies
    # "normal_data": '^.*_normal_data$', # Topics with normal data
    "statistics" : '^.*_statistics$', # Topics with statistics data
    "health_probes": '^.*_HEALTH$', # Topics with health probes data
    "security_topic": "security",    # Training stats for the security manager model
    "dashboard_probes": "DASHBOARD_PROBES",
    "global_metrics": "global_metrics"
}


CLASS_NAMES = ['NORMAL', 'ANOMALY', 'ATTACK']


def decode_array(obj):
    return np.frombuffer(bytes.fromhex(obj["data"]), dtype=obj["dtype"]).reshape(obj["shape"])


def plot_confusion_matrix(cm, title):
    """Interactive Plotly heatmap for a 3×3 confusion matrix."""
    cm_float = cm.astype(float)
    fig = px.imshow(
        cm_float,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        labels=dict(x='Predicted', y='True', color='Count'),
        color_continuous_scale='Blues',
        text_auto='.0f',
        title=title,
    )
    fig.update_xaxes(side='bottom')
    return fig


def _scatter_2d(x, y, color_labels, title):
    """Return a Plotly scatter of 2-D points coloured by string label."""
    return px.scatter(
        {'x': x[:, 0], 'y': x[:, 1], 'label': color_labels},
        x='x', y='y', color='label',
        opacity=0.4,
        title=title,
    )


def plot_results(Y, all_preds, pca_embed, manifold, task_name):
    """Return three interactive Plotly scatter figures (PCA input, manifold labels, manifold preds)."""
    y_sq = Y.squeeze()
    label_names = [EventType(int(v)).name for v in y_sq]
    pred_names  = [EventType(int(v)).name for v in all_preds]

    fig_pca     = _scatter_2d(pca_embed, y_sq, label_names,  f'Input-Space (2D-PCA) {task_name}')
    fig_labels  = _scatter_2d(manifold,  y_sq, label_names,  f'2D-Representation-Space (labels) {task_name}')
    fig_preds   = _scatter_2d(manifold,  y_sq, pred_names,   f'Predictions {task_name}')

    return fig_pca, fig_labels, fig_preds


class KafkaConsumer:
    def __init__(self, parent, kwargs):

        self.parent = parent
        self.is_running = True
        self.current_topics = set()
        self.retry_delay = 1
        self._consumer_closed = False
        self._stop_event = threading.Event()
        # confluent_kafka.Consumer (librdkafka) is not thread-safe: poll(), subscribe(),
        # list_topics() and close() must never run concurrently on the same instance,
        # or the C client's internal state/heap gets corrupted (SIGABRT/segfault).
        self._consumer_lock = threading.Lock()

        def generate_random_string(length=10):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))

        configs = {'bootstrap.servers': kwargs['kafka_broker_url'],
                        'group.id': kwargs['kafka_consumer_group_id']+generate_random_string(7),
                        'auto.offset.reset': kwargs['kafka_auto_offset_reset'],
                        'allow.auto.create.topics': 'true'
                    }

        self.consumer = Consumer(configs)
        self.resubscribe()
        self.topic_update()
        self.consuming_thread = threading.Thread(target=self.consuming_thread_function)
        self.consuming_thread.daemon = True

        self.resubscribe_interval_seconds = int(kwargs['kafka_topic_update_interval_secs'])
        self.resubscription_thread = threading.Thread(target=self.resusbscription_thread_function)
        self.resubscription_thread.daemon = True


    def start(self):
        self.consuming_thread.start()
        self.resubscription_thread.start()


    def stop(self):
        logger = self.parent.logger
        logger.info("KafkaConsumer stop requested — signalling threads to exit.")
        self.is_running = False
        self._stop_event.set()  # wake the resubscription thread immediately

        self.consuming_thread.join(5)
        if self.consuming_thread.is_alive():
            logger.warning("consuming_thread did not stop within 5 s — proceeding anyway.")
        else:
            logger.info("consuming_thread stopped.")

        self.resubscription_thread.join(5)
        if self.resubscription_thread.is_alive():
            logger.warning("resubscription_thread did not stop within 5 s — proceeding anyway.")
        else:
            logger.info("resubscription_thread stopped.")

        logger.info("Closing Kafka consumer...")
        try:
            with self._consumer_lock:
                self.consumer.close()
                self._consumer_closed = True
            logger.info("Kafka consumer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka consumer: {e}")


    def resusbscription_thread_function(self):
        while self.is_running:
            try:
                # Interruptible sleep: _stop_event.set() in stop() wakes this
                # immediately so the thread exits before consumer.close() is called.
                if self._stop_event.wait(timeout=self.resubscribe_interval_seconds):
                    break
                self.topic_update()
            except Exception as e:
                self.parent.logger.error(f"Error in periodic resubscription: {e}")


    def resubscribe(self):
        try:
            with self._consumer_lock:
                if self._consumer_closed:
                    return None
                self.consumer.subscribe(list(topics_dict.values()))
        except KafkaError as e:
            self.parent.logger.error(f"Error subscribing to topics: {e}")
            return None
        self.parent.logger.debug(f"(Re)Started consuming messages from topics: {list(topics_dict.values())}")


    def topic_update(self):
        if self._consumer_closed:
            self.parent.logger.error(
                "RACE DETECTED: resubscription_thread called topic_update() "
                "after consumer was already closed. This can cause a segfault."
            )
            return
        try:
            with self._consumer_lock:
                if self._consumer_closed:
                    return
                available_topics = set(self.consumer.list_topics().topics.keys())
        except Exception as e:
            self.parent.logger.error(f"topic_update: list_topics() raised {type(e).__name__}: {e}")
            return
        new_topics = available_topics - self.current_topics
        self.current_topics = available_topics
        if len(new_topics) > 0:
            self.parent.logger.debug(f"New topics: {list(new_topics)}; Number of available topics: {len(self.current_topics)}")
            self.resubscribe()


    def deserialize_message(self, msg):
        try:
            message_value = json.loads(msg.value().decode('utf-8'))
            return message_value
        except json.JSONDecodeError as e:
            self.parent.logger.error(f"Error deserializing message: {e}")
            return None


    def consuming_thread_function(self):

        while self.is_running:
            try:
                with self._consumer_lock:
                    if self._consumer_closed:
                        break
                    msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        self.parent.logger.debug(f"End of partition reached: {msg.error()}")
                    elif (msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART) and \
                        (msg.error().str().split(': ')[1] in list(topics_dict.values())):
                            self.parent.logger.info(f"Note: {msg.error().str()}")
                    else:
                        self.parent.logger.error(f"Consumer error: {msg.error()}")
                    continue

                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.parent.logger.debug(f"Processing message from topic {msg.topic()}")
                    if 'statistics' in msg.topic():
                        self._handle_statistics_message(msg.topic(), deserialized_data)
                else:
                    self.parent.logger.warning("Deserialized message is None")

                self.retry_delay = 1
            except Exception as e:
                self.parent.logger.error(f"Error while reading message: {e}")
                self.parent.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.retry_delay = min(self.retry_delay * 2, 60)


    def _handle_statistics_message(self, topic, data):
        """Route a statistics-topic message to the appropriate W&B logging path."""
        vehicle_name = topic.split('_')[0]

        if 'adv_eval_accuracy' in data:
            # ── Gaussian-noise adversarial evaluation ─────────────────────────────
            # Plot keys are only present when the producer decided this round
            # should also include scatter plots / confusion matrices.
            if 'visual_eval_X' in data:
                fig_pca, fig_labels, fig_preds = plot_results(
                    decode_array(data['visual_eval_y']),
                    decode_array(data['visual_eval_preds']),
                    decode_array(data['visual_eval_X']),
                    decode_array(data['visual_eval_manifold']),
                    vehicle_name + ' manifold',
                )
                self.parent.push_to_wandb(key=f"{vehicle_name}_manifold_pca",    value=fig_pca)
                self.parent.push_to_wandb(key=f"{vehicle_name}_manifold_labels", value=fig_labels)
                self.parent.push_to_wandb(key=f"{vehicle_name}_manifold_preds",  value=fig_preds)
                self.parent.push_to_wandb(
                    key=f"{vehicle_name}_adv_eval_confusion_matrix",
                    value=plot_confusion_matrix(
                        decode_array(data['adv_eval_confusion_matrix']).astype(int),
                        f"{vehicle_name} Gaussian adv eval",
                    ))
            self.parent.push_to_wandb(
                key=topic,
                value={
                    'adv_eval_accuracy':  data['adv_eval_accuracy'],
                    'adv_eval_precision': data['adv_eval_precision'],
                    'adv_eval_recall':    data['adv_eval_recall'],
                    'adv_eval_f1':        data['adv_eval_f1'],
                    'adv_eval_macro_f1':  data['adv_eval_macro_f1'],
                })

        elif any(k.startswith('hsja_adv_eval/') for k in data):
            # ── HopSkipJump decision-based adversarial evaluation ─────────────────
            # Left panel  : PCA of original (clean) feature space.
            # Centre panel: adversarial examples in manifold space, coloured by TRUE label.
            # Right panel : adversarial examples in manifold space, coloured by PREDICTED label.
            # Plot keys are only present when the producer decided this round
            # should also include scatter plots / confusion matrices.
            if 'hsja_visual_eval_X' in data:
                fig_pca, fig_labels, fig_preds = plot_results(
                    decode_array(data['hsja_visual_eval_y']),
                    decode_array(data['hsja_visual_eval_preds']),
                    decode_array(data['hsja_visual_eval_X']),
                    decode_array(data['hsja_visual_eval_manifold']),
                    vehicle_name + ' HSJA',
                )
                self.parent.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_pca",    value=fig_pca)
                self.parent.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_labels", value=fig_labels)
                self.parent.push_to_wandb(key=f"{vehicle_name}_hsja_manifold_preds",  value=fig_preds)
                if 'hsja_adv_eval_confusion_matrix' in data:
                    self.parent.push_to_wandb(
                        key=f"{vehicle_name}_hsja_adv_eval_confusion_matrix",
                        value=plot_confusion_matrix(
                            decode_array(data['hsja_adv_eval_confusion_matrix']).astype(int),
                            f"{vehicle_name} HSJA adv eval",
                        ))
            # Scalar metrics: '/' creates the 'hsja_adv_eval' sub-section in W&B.
            hsja_scalars = {k: v for k, v in data.items() if k.startswith('hsja_adv_eval/')}
            if hsja_scalars:
                self.parent.push_to_wandb(key=topic, value=hsja_scalars)

        else:
            # ── Regular per-epoch training / online monitoring statistics ──────────
            if 'online_confusion_matrix' in data:
                self.parent.push_to_wandb(
                    key=f"{vehicle_name}_online_confusion_matrix",
                    value=plot_confusion_matrix(
                        decode_array(data.pop('online_confusion_matrix')).astype(int),
                        f"{vehicle_name} online monitoring",
                    ))
            self.parent.push_to_wandb(key=topic, value=data)
