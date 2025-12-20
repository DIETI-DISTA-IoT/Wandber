import wandb
import logging
from kafka_consumer import KafkaConsumer
import signal 
from OpenFAIR.container_api import ContainerAPI
import threading 
from preprocessing import HealthProbesBuffer
from brain import Brain
from communication import SMMetricsReporter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from confluent_kafka.admin import AdminClient
from preprocessing import GenericBuffer
import torch
import string
import random
import time
import pickle
from confluent_kafka import Consumer, KafkaError
import json
import requests
from modules import MLP
from reporting import WeightsReporter, GlobalMetricsReporter
from aggregation import federated_averaging, fed_yogi
import sys

MANAGER = "MANAGER"
WANDBER = "WANDBER"
SECURITY_MANAGER = "SECURITY_MANAGER"
FEDERATED_LEARNING = "FEDERATED_LEARNING"
HEALTHY = "HEALTHY"
INFECTED = "INFECTED"

aggregation_functions = {
    "fedavg": federated_averaging,
    "fedyogi": fed_yogi,
    "fedprox": None,
    "fedsgd": None
    }


class Wandber:


    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=str(args['logging_level']).upper())
        self.logger = logging.getLogger(WANDBER)
        self.logger.setLevel(args['logging_level'].upper())
        self.logger.debug("Initializing wandb")
        self.wandb_mode = ("online" if args['online'] else "disabled")
        wandb.init(
            project=args['project_name'],
            mode=self.wandb_mode,
            name=args['run_name'],
            config=args
        )
        self.logger.debug(f"Wandb initialized in {self.wandb_mode} mode")
        self.step = 0
        self.kafka_consumer = KafkaConsumer(
            parent=self,
            kwargs=args
        )
        self.kafka_consumer.start()
    

    def graceful_shutdown(self):
        self.kafka_consumer.stop()
        self.close_wandb()


    def push_to_wandb(self, key, value, step=None, commit=True):
        # self.logger.debug(f"Pushing {key} to wandb")
        wandb.log(
            {key: value}, 
            step=(step if step is not None else self.step), 
            commit=commit)
        if step is None:
            self.step += 1


    def close_wandb(self):
        wandb.finish()
        self.logger.debug("Wandb closed correctly.")


class SecurityManager:

    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=str(args['logging_level']).upper())
        self.logger = logging.getLogger(SECURITY_MANAGER)
        self.logger.setLevel(args['logging_level'].upper())
        self.logger.debug("Initializing security manager")
        self.dashboard_endpoint = args['dashboard_endpoint']

        self.health_records_received = 0
        self.victim_records_received = 0
        self.normal_records_received = 0
        self.mitigation_reward = 0
        self.online_batch_labels = []
        self.online_batch_preds = []
        
        self.mitigation = args['mitigation']
        self.true_positive_reward = args['true_positive_reward']
        self.false_positive_reward = args['false_positive_reward']
        self.true_negative_reward = args['true_negative_reward']
        self.false_negative_reward = args['false_negative_reward']
        self.mitigation_reward = 0
        self.brain = Brain(**args)
        self.metrics_reporter = SMMetricsReporter(**args)
        self.victim_buffer = HealthProbesBuffer(args['buffer_size'], label=1)
        self.normal_buffer = HealthProbesBuffer(args['buffer_size'], label=0)
        
        self.resubscribe_interval_seconds = int(args['kafka_topic_update_interval_secs'])
        self.resusbscription_thread = threading.Thread(target=self.resubscribe, daemon=True)
        self.logger.debug("Security manager initialized")

        self.stats_consuming_thread=threading.Thread(target=self.consume_health_data, kwargs=args, daemon=True)
        self.training_thread=threading.Thread(target=self.train_model, kwargs=args)
        
        self.stop_threads = False

        self.stats_consuming_thread.start()
        self.training_thread.start()
        self.resusbscription_thread.start()
        self.logger.info("Security manager started!")


    def get_status_from_dashboard(self, vehicle_name):
        url = f"http://{self.dashboard_endpoint}/vehicle-status"
        data = {"vehicle_name": vehicle_name}
        response = requests.post(url, json=data)
        self.logger.debug(f"Vehicle-status Response Status Code: {response.status_code}")
        self.logger.debug(f"Vehicle-status Response Body: {response.text}")
        return response.text


    def resubscribe(self):
        while not self.stop_threads:
            try:
                # Wait for a certain interval before resubscribing
                time.sleep(self.resubscribe_interval_seconds)
                self.subscribe_to_topics('^.*_HEALTH$')
            except Exception as e:
                self.logger.error(f"Error in periodic resubscription: {e}")


    def train_model(self, **kwargs):

        batch_size = int(kwargs.get('batch_size'))
        epoch_size = int(kwargs.get('epoch_size'))
        save_model_freq_epochs = int(kwargs.get('save_model_freq_epochs'))

        while not self.stop_threads:
            batch_feats = None
            batch_labels = None
            batch_preds = None
            do_train_step = False
            batch_loss = 0

            normal_feats, normal_labels = self.normal_buffer.sample(batch_size)
            victim_feats, victim_labels = self.victim_buffer.sample(batch_size)

            if len(normal_feats) > 0:
                batch_feats = normal_feats
                do_train_step = True
                batch_labels = normal_labels

            if len(victim_feats) > 0:
                do_train_step = True
                batch_feats = (victim_feats if batch_feats is None else torch.vstack((batch_feats, victim_feats)))
                batch_labels = (victim_labels if batch_labels is None else torch.vstack((batch_labels, victim_labels)))

            if do_train_step:
                batch_counter += 1
                batch_preds, loss = self.brain.train_step(batch_feats, batch_labels)


                # convert bath_preds to binary using pytorch:
                batch_preds = (batch_preds > 0.5).float()
                batch_loss += loss
                batch_accuracy = accuracy_score(batch_labels, batch_preds)
                batch_precision = precision_score(batch_labels, batch_preds, zero_division=0)
                batch_recall = recall_score(batch_labels, batch_preds, zero_division=0)
                batch_f1 = f1_score(batch_labels, batch_preds, zero_division=0)


                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                epoch_precision += batch_precision
                epoch_recall += batch_recall
                epoch_f1 += batch_f1

                if batch_counter % epoch_size == 0:
                    epoch_counter += 1

                    epoch_loss /= epoch_size
                    epoch_accuracy /= epoch_size
                    epoch_precision /= epoch_size
                    epoch_recall /= epoch_size
                    epoch_f1 /= epoch_size

                    self.metrics_reporter.report({
                        'total_loss': epoch_loss,
                        'accuracy': epoch_accuracy,
                        'precision': epoch_precision,
                        'recall': epoch_recall,
                        'f1': epoch_f1,
                        'diagnostics_processed': self.normal_records_received,
                        'anomalies_processed': self.victim_records_received})

                    epoch_loss = epoch_accuracy = epoch_precision = epoch_recall = epoch_f1 = 0

                    if epoch_counter % save_model_freq_epochs == 0:
                        self.logger.debug(f"Training Epoch {epoch_counter}: loss={epoch_loss}, accuracy={epoch_accuracy}, precision={epoch_precision}, recall={epoch_recall}, f1={epoch_f1}")
                        model_path = kwargs.get('model_saving_path', 'default_sm_model.pth')
                        self.logger.info(f"Saving model after {epoch_counter} epochs as {model_path}.")
                        self.brain.save_model()

            time.sleep(int(kwargs.get('training_freq_seconds')))


    def create_consumer(self,**kwargs):
        def generate_random_string(length=10):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for i in range(length))
        # Kafka consumer configuration
        conf_cons = {
            'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
            'group.id': kwargs.get('kafka_consumer_group_id')+generate_random_string(7),  # Consumer group ID for message offset tracking
            'auto.offset.reset': kwargs.get('kafka_auto_offset_reset')  # Start reading from the earliest message if no offset is present
        }
        return Consumer(conf_cons)


    def deserialize_message(self, msg):
        try:
            # Decode the message and deserialize it into a Python dictionary
            message_value = json.loads(msg.value().decode('utf-8'))
            self.logger.debug(f"received message from topic [{msg.topic()}]")
            return message_value
        except json.JSONDecodeError as e:
            self.logger.error(f"Error deserializing message: {e}")
            return None


    def store_datum_on_buffer(self, current_vehicle_status, msg):

        if current_vehicle_status == INFECTED:
            self.victim_buffer.add(msg)
            self.victim_records_received += 1
        else:
            self.normal_buffer.add(msg)
            self.normal_records_received += 1


    def online_classification(self, msg):
        self.brain.model.eval()
        with self.brain.model_lock, torch.no_grad():
            x = torch.tensor(list(msg.values()), dtype=torch.float32)
            y_pred = self.brain.model(x)
            y_pred = (y_pred > 0.5).float()
            return y_pred
        

    def send_attack_mitigation_request(self, vehicle_name):
        url = f"http://{self.dashboard_endpoint}/stop-attack"
        data = {"vehicle_name": vehicle_name, "origin": "AI"}
        response = requests.post(url, json=data)
        try:
            response_json = response.json()
            self.logger.debug(f"Mitigate-attack Response JSON: {response_json}")
            self.metrics_reporter.report({'mitigation_time': response_json.get('mitigation_time', 0)})
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from response: {e}")
            response_json = {}


    def mitigation_and_rewarding(self, prediction, current_label, vehicle_name):
        if prediction == 1:
            if prediction == current_label:
                # True positive.
                if self.mitigation:
                    self.send_attack_mitigation_request(vehicle_name)
                self.mitigation_reward += self.true_positive_reward
            else:
                # False positive
                self.mitigation_reward += self.false_positive_reward
        else:
            if prediction == current_label:
                # True negative
                self.mitigation_reward += self.true_negative_reward
            else:
                # False negative
                self.mitigation_reward += self.false_negative_reward


    def process_message(self,topic, msg):

        self.logger.debug(f"Processing message from topic [{topic}]")
        assert topic.endswith("_HEALTH"), f"Unexpected topic {topic}"
        
        self.health_records_received += 1

        vehicle_name = topic.split('_')[0]
        current_vehicle_status = self.get_status_from_dashboard(vehicle_name)
        current_label = 1 if current_vehicle_status == INFECTED else 0
        self.online_batch_labels.append(torch.tensor(current_label).float())

        self.store_datum_on_buffer(current_vehicle_status, msg)
        
        prediction = self.online_classification(msg)
        self.online_batch_preds.append(prediction)

        self.mitigation_and_rewarding(prediction, current_label, vehicle_name)
            
        if self.health_records_received % 50 == 0:
            self.logger.info(f"Received {self.health_records_received} health records: {self.victim_records_received} victims, {self.normal_records_received} normal.")
            self.online_batch_accuracy = accuracy_score(self.online_batch_labels, self.online_batch_preds)
            self.online_batch_precision = precision_score(self.online_batch_labels, self.online_batch_preds, zero_division=0)
            self.online_batch_recall = recall_score(self.online_batch_labels, self.online_batch_preds, zero_division=0)
            self.online_batch_f1 = f1_score(self.online_batch_labels, self.online_batch_preds, zero_division=0)
            self.metrics_reporter.report({
                        'online_accuracy': self.online_batch_accuracy,
                        'online_precision': self.online_batch_precision,
                        'online_recall': self.online_batch_recall,
                        'online_f1': self.online_batch_f1,
                        'mitigation_reward': self.mitigation_reward
                        })
            self.logger.debug(f"Online metrics: accuracy={self.online_batch_accuracy}," + \
                        f"precision={self.online_batch_precision}, " + \
                        f"recall={self.online_batch_recall}, " + \
                        f"f1={self.online_batch_f1}" + \
                        f"mitigation_reward={self.mitigation_reward}")
            
            self.online_batch_labels = []
            self.online_batch_preds = []
            self.mitigation_reward = 0


    def consume_health_data(self, **kwargs):

        self.consumer = self.create_consumer(**kwargs)

        self.subscribe_to_topics('^.*_HEALTH$')

        try:
            while not self.stop_threads:
                msg = self.consumer.poll(5.0)  # Poll per 1 secondo
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        self.logger.info(f"End of partition reached for {msg.topic()}")
                    else:
                        self.logger.error(f"consumer error: {msg.error()}")
                    continue

                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.process_message(msg.topic(), deserialized_data)

        except KeyboardInterrupt:
            self.logger.info(f"consumer interrupted by user.")
        except Exception as e:
            self.logger.error(f" error in consumer {e}")
        finally:
            self.consumer.close()
            self.logger.info(f"consumer closed.")


    def subscribe_to_topics(self, topic_regex):

        self.consumer.subscribe([topic_regex])
        self.logger.debug(f"(re)subscribed to health topics.")


    def graceful_shutdown(self):
        self.stop_threads = True
        if self.resusbscription_thread:
            self.resusbscription_thread.join()
        if self.stats_consuming_thread:
            self.stats_consuming_thread.join()
        if self.training_thread:
            self.training_thread.join()
        self.logger.info(f"Security manager stopped.")


class FederatedLearningManager:


    def __init__(self, args):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=str(args['logging_level']).upper())
        self.logger = logging.getLogger(FEDERATED_LEARNING)
        self.logger.info("Initializing federated learning manager")
        self.aggregation_interval_secs = args['aggregation_interval_secs']

        self.global_model = MLP(**args)
        self.global_model.initialize_weights(args['initialization_strategy'])
        self.logger.info(f"Global model initialized using {args['initialization_strategy']} initialization.")

        self.admin_client = AdminClient({'bootstrap.servers': args['kafka_broker_url']})
        self.vehicle_weights_topics = self.check_vehicle_weights_topics(args)

        self.weights_buffer = self.create_weights_buffer(**args)
        self.weights_reporter = WeightsReporter(self.logger, **args)
        self.global_metrics_reporter = GlobalMetricsReporter(self.logger, **args)

        # self.eval_feats, self.eval_labels = self.load_eval_df(**args)
        self.logger.info(f"Starting FL with {len(self.vehicle_weights_topics)}" + \
                         f" for vehicles: {self.vehicle_weights_topics}")
        self.stop_threads = False

        self.consuming_thread = threading.Thread(
            target=self.consume_weights_data, 
            kwargs=args, 
            daemon=True
            )
        
        self.consuming_thread.start()

        if self.aggregation_interval_secs > 0:
            # create a thread to aggregate the weights each aggregation_interval_secs:
            self.aggregation_thread = threading.Thread(
                target=self.aggregate_weights_periodically, 
                kwargs=args)
            self.aggregation_thread.start()
    

    def graceful_shutdown(self):
        self.stop_threads = True
        if self.consuming_thread:
            self.consuming_thread.join()
        if self.aggregation_thread:
            self.aggregation_thread.join()
        self.logger.info(f"Federated learning manager stopped.")


    def aggregate_weights_periodically(self, **kwargs):
        while not self.stop_threads:
            time.sleep(kwargs.get('aggregation_interval_secs'))
            self.aggregate_weights(**kwargs)
            # self.evaluate_new_model()


    def evaluate_new_model(self):
        epoch_accuracy = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        self.logger.debug("Evaluating new model...")
        self.global_model.eval()
        len_eval_ds = len(self.eval_feats)
        with torch.no_grad():
            for batch_feats, batch_labels in zip(self.eval_feats, self.eval_labels): 
                batch_preds = self.global_model(batch_feats)
                batch_preds = (batch_preds > 0.5).float()
                epoch_accuracy += accuracy_score(batch_labels, batch_preds)
                epoch_precision += precision_score(batch_labels, batch_preds, zero_division=0)
                epoch_recall += recall_score(batch_labels, batch_preds, zero_division=0)
                epoch_f1 += f1_score(batch_labels, batch_preds, zero_division=0)
            
        epoch_accuracy /= len_eval_ds
        epoch_precision /= len_eval_ds
        epoch_recall /= len_eval_ds
        epoch_f1 /= len_eval_ds
        self.logger.info(f"Eval Accuracy: {epoch_accuracy}, Precision: {epoch_precision}, Recall: {epoch_recall}, F1: {epoch_f1}")
        self.global_metrics_reporter.report_metrics({'accuracy': epoch_accuracy, 'precision': epoch_precision, 'recall': epoch_recall, 'f1': epoch_f1})
        

    def consume_weights_data(self, **kwargs):

        consumer = self.create_consumer(**kwargs)

        consumer.subscribe(self.vehicle_weights_topics)
        self.logger.info(f"will start consuming {self.vehicle_weights_topics}")

        try:
            while not self.stop_threads:
                msg = consumer.poll(5.0)  
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        self.logger.info(f"End of partition reached for {msg.topic()}")
                    else:
                        self.logger.error(f"consumer error: {msg.error()}")
                    continue

                deserialized_data = self.deserialize_message(msg)
                if deserialized_data:
                    self.process_message(msg.topic(), deserialized_data, **kwargs)

        except KeyboardInterrupt:
            self.logger.info(f" Consumer interrupted by user.")
        except Exception as e:
            self.logger.error(f" Error in consumer: {e}")
        finally:
            consumer.close()
            self.logger.info(f" Consumer closed.")


    def create_consumer(self, **kwargs):
        # Kafka consumer configuration
        conf_cons = {
            'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
            'group.id': kwargs.get('kafka_consumer_group_id'),  # Consumer group ID
            'auto.offset.reset': kwargs.get('kafka_auto_offset_reset')
        }
        return Consumer(conf_cons)


    def deserialize_message(self, msg):

        try:
            # Decode the message and deserialize it into a Python dictionary
            message_value = pickle.loads(msg.value())
            self.logger.debug(f"received message from topic [{msg.topic()}]")
            return message_value
        except json.JSONDecodeError as e:
            self.logger.error(f"Error deserializing message: {e}")
            return None


    def aggregate_weights(self, **kwargs):

        # check if we have at least one element in each buffer:
        if all([len(buffer) > 0 for buffer in self.weights_buffer.values()]):
            self.logger.info(f"Aggregating the weights from {len(self.weights_buffer)} vehicles.")
            aggregation_function = aggregation_functions[kwargs.get('aggregation_strategy')]

            for buffer in self.weights_buffer.values():
                candidate_state_dict = buffer.get()
                if any(torch.isnan(param).any() for param in candidate_state_dict.values()):
                    self.logger.error(f"Candidate weights from {buffer.label} contain NaNs. Skipping update.")
                    # buffer.pop()
                    return
            
            if aggregation_function is federated_averaging:
                aggregated_state_dict = aggregation_function(
                    self.global_model.state_dict(), 
                    [buffer.get() for buffer in self.weights_buffer.values()])
            else:
                aggregated_state_dict = aggregation_function(
                    self.global_model.state_dict(), 
                    [buffer.get() for buffer in self.weights_buffer.values()], **kwargs)
            
            # Check if the aggregated state dict has no NaNs
            if any(torch.isnan(param).any() for param in aggregated_state_dict.values()):
                self.logger.error("Aggregated state dict contains NaNs. Skipping update.")
                return
            # pop the oldest element from each one of the buffers
            for buffer in self.weights_buffer.values():
                buffer.pop()
            self.global_model.load_state_dict(aggregated_state_dict)
            self.weights_reporter.push_weights(self.global_model.state_dict())
        else:
            self.logger.info(f"Waiting for more data to aggregate the weights.")



    def process_message(self, topic, msg, **kwargs):

        self.weights_buffer[topic].add(msg)

        if kwargs.get('aggregation_interval_secs') == 0:
            self.aggregate_weights(**kwargs)


    def create_weights_buffer(self, **kwargs):
        weights_buffer = {}
        for topic in self.vehicle_weights_topics:
            weights_buffer[topic] = GenericBuffer(size=kwargs.get('weights_buffer_size', 3), label=topic)
        return weights_buffer


    def check_vehicle_weights_topics(self, args):
        existing_topics = self.admin_client.list_topics(timeout=10).topics.keys()
        # get all topics ending with "_weights"
        vehicle_topics = [topic for topic in existing_topics if topic.endswith("_weights") and topic != "global_weights"]
        self.logger.debug("Found the following vehicle topics: %s", vehicle_topics)
        return vehicle_topics


class ManagerAPI(ContainerAPI):

    def __init__(self, port: int = 5000):
        super().__init__(
            container_type='manager',
            container_name='manager',
            port=port
            )
        self.wandber_instance = None
        self.sm_instance = None
        self.fl_instance = None
        
    
    def handle_command(self, command, params):
        if command == 'start_wandb':
            self.wandber_instance = Wandber(params)
            return "Succesfully started wandb"
        elif command == 'stop_wandb':
            if self.wandber_instance is not None:
                self.wandber_instance.graceful_shutdown()
                return "Succesfully stopped wandb"
            else:
                return "WandB is not running"
        elif command == 'start_security_manager':
            self.sm_instance = SecurityManager(params)
            return "Succesfully started security manager"
        elif command == 'stop_security_manager':
            if self.sm_instance is not None:
                self.sm_instance.graceful_shutdown()
                return "Succesfully stopped security manager"
            else:
                return "Security manager is not running"
        elif command == "start_mitigation":
            if self.sm_instance is None:
                return "Security manager is not running"
            self.sm_instance.mitigation = True
            self.logger.info(f"Mitigation is {self.sm_instance.mitigation}")
            return "Succesfully started mitigation"
        elif command == "stop_mitigation":
            if self.sm_instance is None:
                return "Security manager is not running"
            self.logger.info(f"Mitigation is {self.sm_instance.mitigation}")
            self.sm_instance.mitigation = False
            return "Succesfully stopped mitigation"
        elif command == "start_federated_learning":
            if self.fl_instance is None:
                self.fl_instance = FederatedLearningManager(params)
            return "Succesfully started federated learning"
        elif command == "stop_federated_learning":
            if self.fl_instance is not None:
                self.fl_instance.graceful_shutdown()
                return "Succesfully stopped federated learning"
            else:
                return "Federated learning is not running"
        else:
            return "Unrecognized command"


def signal_handler(sig, frame):
    global api

    print(f"{MANAGER}: Received signal {sig}. Gracefully stopping wandb and its consumer threads.")
    if api.wandber_instance is not None:
        api.wandber_instance.graceful_shutdown()
    if api.sm_instance is not None:
        api.sm_instance.graceful_shutdown()
    if api.fl_instance is not None:
        api.fl_instance.graceful_shutdown()
    sys.exit(0)


def main():
    global api
    """
    parser = argparse.ArgumentParser(description='Wandb reporter process for Open FAIR.')
    parser.add_argument('--logging_level', default='INFO' ,type=str, help='Logging level')
    parser.add_argument('--project_name', type=str, default="OPEN_FAIR", help='Wandb Project name')
    parser.add_argument('--run_name', type=str, default="Some run", help='Wandb run name')
    parser.add_argument('--online', action='store_true', help='Send wand metrics to the public wandb cloud')
    parser.add_argument('--kafka_broker_url', type=str, default='kafka:9092', help='Kafka broker URL')
    parser.add_argument('--kafka_consumer_group_id', type=str, default=WANDBER, help='Kafka consumer group ID')
    parser.add_argument('--kafka_auto_offset_reset', type=str, default='earliest', help='Start reading messages from the beginning if no offset is present')
    parser.add_argument('--kafka_topic_update_interval_secs', type=int, default=30, help='Topic update interval for the kafka reader')
    args = parser.parse_args()
    wandber = Wandber(args)
    """
    api = ManagerAPI()
    api.run()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

if __name__ == "__main__":
    main()
    
