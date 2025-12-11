from confluent_kafka import Consumer, KafkaError
import time
import json
import threading
import numpy as np
import wandb
from matplotlib import pyplot as plt
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


def decode_array(obj):
    return np.frombuffer(bytes.fromhex(obj["data"]), dtype=obj["dtype"]).reshape(obj["shape"])


def plot_results(Y, all_preds, pca_embed, manifold, task_name):

        fig, axes = plt.subplots(1, 3, figsize=(20, 4))

        colors = ['r', 'g', 'b']

        # First subplot
        ax = axes[0]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(pca_embed[mask, 0], pca_embed[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'Input-Space (2D-PCA) {task_name}')
        ax.legend()

        # Second subplot
        ax = axes[1]
        for eventype in EventType:
            mask = Y.squeeze() == eventype.value
            ax.scatter(manifold[mask, 0], manifold[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'2D-Representation-Space (labels) {task_name}')
        ax.legend()
        

        # Third subplot
        ax = axes[2]
        for eventype in EventType:
            mask = all_preds == eventype.value
            ax.scatter(manifold[mask, 0], manifold[mask, 1],
                    c=colors[eventype.value], s=15, alpha=0.1, label=eventype.name)
        ax.set_title(f'Predictions {task_name}')
        ax.legend()

        return fig


class KafkaConsumer:
    def __init__(self, parent, kwargs):

        self.parent = parent
        self.is_running = True
        self.current_topics = set()
        self.retry_delay = 1

        def generate_random_string(length=10):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))

        configs = {'bootstrap.servers': kwargs['kafka_broker_url'],  # Kafka broker URL
                        'group.id': kwargs['kafka_consumer_group_id']+generate_random_string(7),  # Consumer group for offset management
                        'auto.offset.reset': kwargs['kafka_auto_offset_reset'],  # Start reading messages from the beginning if no offset is present
                        'allow.auto.create.topics': 'true'  # crucial for topic updating
                    }
        
        self.consumer = Consumer(configs)
        self.resubscribe()
        self.topic_update()
        self.consuming_thread = threading.Thread(target=self.consuming_thread_function)
        self.consuming_thread.daemon = True

        # Thread for periodic resubscription
        self.resubscribe_interval_seconds = int(kwargs['kafka_topic_update_interval_secs'])
        self.resubscription_thread = threading.Thread(target=self.resusbscription_thread_function)
        self.resubscription_thread.daemon = True


    def start(self):
        """
        Start both reading and resubscription threads
        """
        self.consuming_thread.start()
        self.resubscription_thread.start()


    def stop(self):
        """
        Gracefully stop the consumer and its threads
        """
        self.is_running = False
        self.consuming_thread.join(1)
        self.resubscription_thread.join(1)
        self.consumer.close()


    def resusbscription_thread_function(self):
        """
        Periodically Kafka topics update.
        This method runs in a separate thread.
        """
        while self.is_running:
            try:
                # Wait for a certain interval before resubscribing
                time.sleep(self.resubscribe_interval_seconds)
                self.topic_update()
            except Exception as e:
                self.parent.logger.error(f"Error in periodic resubscription: {e}")


    def resubscribe(self):
        try:
            self.consumer.subscribe(list(topics_dict.values()))
        except KafkaError as e:
            self.parent.logger.error(f"Error subscribing to topics: {e}")
            return None
        self.parent.logger.debug(f"(Re)Started consuming messages from topics: {list(topics_dict.values())}")


    def topic_update(self):
        available_topics = set(self.consumer.list_topics().topics.keys())
        new_topics = available_topics - self.current_topics
        self.current_topics = available_topics
        if len(new_topics) > 0:
            self.parent.logger.debug(f"New topics: {list(new_topics)}; Number of available topics: {len(self.current_topics)}")
            self.resubscribe()


    def deserialize_message(self, msg):
        try:
            # Decode the message value from bytes to string and parse JSON
            message_value = json.loads(msg.value().decode('utf-8'))
            # self.parent.logger.debug(f"Received message from topic {msg.topic()}")
            return message_value
        except json.JSONDecodeError as e:
            self.parent.logger.error(f"Error deserializing message: {e}")
            return None
    

    def consuming_thread_function(self):
        
        while self.is_running:
            try:
                msg = self.consumer.poll(1.0)  # Poll for new messages with a timeout of 1 second
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

                # Deserialize the message and process it
                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.parent.logger.debug(f"Processing message from topic {msg.topic()}")
                    if 'statistics' in msg.topic():
                        if "visual_eval_X" in deserialized_data:
                            vehicle_name = msg.topic().split('_')[0]
                            visual_eval_X = decode_array(deserialized_data['visual_eval_X'])
                            visual_eval_y = decode_array(deserialized_data['visual_eval_y'])
                            visual_eval_preds = decode_array(deserialized_data['visual_eval_preds'])
                            visual_eval_manifold = decode_array(deserialized_data['visual_eval_manifold'])
                            manifold_fig = plot_results(visual_eval_y, visual_eval_preds, visual_eval_X, visual_eval_manifold, vehicle_name+' manifold')
                            wandb_manifold_fig = wandb.Image(manifold_fig)
                            self.parent.push_to_wandb(
                                key=f"{vehicle_name}_manifold_plot",
                                value=wandb_manifold_fig)
                            plt.close(manifold_fig)   # <-- THIS FIXES THE FIGURE LEAK
                        else:     
                            self.parent.push_to_wandb(
                                key=msg.topic(), 
                                value=deserialized_data)
                else:
                    self.parent.logger.warning("Deserialized message is None")

                self.retry_delay = 1  # Reset retry delay on success
            except Exception as e:
                self.parent.logger.error(f"Error while reading message: {e}")
                self.parent.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.retry_delay = min(self.retry_delay * 2, 60)  # Exponential backoff, max 60 seconds