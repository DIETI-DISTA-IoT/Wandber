import argparse
import logging
import threading
import json
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.admin import AdminClient
from aggregation import federated_averaging, FedYogi, fed_median, fed_prox
import pickle
from modules import MLP
from preprocessing import GenericBuffer
import time
from reporting import WeightsReporter, GlobalMetricsReporter
import signal
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


FEDERATED_LEARNING = "fed_learning"

# FedYogi is a stateful class instance so its moment buffers survive across rounds.
_fed_yogi_instance = FedYogi()

aggregation_functions = {
    "fedavg": federated_averaging,
    "fedyogi": _fed_yogi_instance,
    "fedmedian": fed_median,
    "fedprox": fed_prox,
    "fedsgd": None,
}


def create_consumer(**kwargs):
    # Kafka consumer configuration
    conf_cons = {
        'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
        'group.id': kwargs.get('kafka_consumer_group_id'),  # Consumer group ID
        'auto.offset.reset': kwargs.get('kafka_auto_offset_reset')
    }
    return Consumer(conf_cons)



def check_vehicle_topics(**kwargs):

    admin_client = AdminClient({'bootstrap.servers': kwargs.get('kafka_broker_url')})
    existing_topics = admin_client.list_topics(timeout=10).topics.keys()
    # get all topics ending with "_weights"
    vehicle_topics = [topic for topic in existing_topics if topic.endswith("_weights") and topic != "global_weights"]
    logger.debug("Found the following vehicle topics: %s", vehicle_topics)
    return vehicle_topics


def deserialize_message(msg):

    try:
        # Decode the message and deserialize it into a Python dictionary
        message_value = pickle.loads(msg.value())
        logger.debug(f"received message from topic [{msg.topic()}]")
        return message_value
    except json.JSONDecodeError as e:
        logger.error(f"Error deserializing message: {e}")
        return None


def init_global_model(**kwargs):
    initialization_strategy = kwargs.get('initialization_strategy')
    global_model.initialize_weights(initialization_strategy)
    # Reset FedYogi state when the global model is re-initialised so that
    # stale moments from a previous run don't corrupt the new one.
    _fed_yogi_instance.reset()
    logger.info(f"Global model initialized using {initialization_strategy} initialization.")
    

def process_message(topic, msg, **kwargs):
    global weights_buffer

    weights_buffer[topic].add(msg)

    if kwargs.get('aggregation_interval_secs') == 0:
        aggregate_weights(**kwargs)
        

def aggregate_weights_periodically(**kwargs):
    while not stop_threads:
        time.sleep(kwargs.get('aggregation_interval_secs'))
        aggregate_weights(**kwargs)
        if mode == 'OF':
            evaluate_new_model()


def dict_to_tensor(data_dict):
    
    uncampled_values = [ (value  if isinstance(value, (int, float)) and not np.isnan(value) else 0.0) for value in data_dict.values() ]
    clampled_values = [ max(min(value, 3000000), -4000) for value in uncampled_values ]
    # Convert the list of values to a PyTorch tensor
    tensor = torch.tensor(clampled_values, dtype=torch.float32)
    return tensor


def evaluate_new_model():
    global epoch_accuracy, epoch_precision, epoch_recall, epoch_f1
    epoch_accuracy = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    logger.debug("Evaluating new model...")
    global_model.eval()
    with torch.no_grad():
        for batch_feats, batch_labels in zip(eval_feats, eval_labels): 
            batch_preds = global_model(batch_feats)
            batch_preds = (batch_preds > 0.5).float()
            epoch_accuracy += accuracy_score(batch_labels, batch_preds)
            epoch_precision += precision_score(batch_labels, batch_preds, zero_division=0)
            epoch_recall += recall_score(batch_labels, batch_preds, zero_division=0)
            epoch_f1 += f1_score(batch_labels, batch_preds, zero_division=0)
        
    epoch_accuracy /= len(eval_feats)
    epoch_precision /= len(eval_feats)
    epoch_recall /= len(eval_feats)
    epoch_f1 /= len(eval_feats)
    logger.info(f"Eval Accuracy: {epoch_accuracy}, Precision: {epoch_precision}, Recall: {epoch_recall}, F1: {epoch_f1}")
    global_metrics_reporter.report_metrics({'accuracy': epoch_accuracy, 'precision': epoch_precision, 'recall': epoch_recall, 'f1': epoch_f1})
    

def aggregate_weights(**kwargs):
    global global_model, weights_buffer

    # check if we have at least one element in each buffer:
    if all([len(buffer) > 0 for buffer in weights_buffer.values()]):
        logger.debug(f"Aggregating the weights from {len(weights_buffer)} vehicles.")
        aggregation_function = aggregation_functions[kwargs.get('aggregation_strategy')]

        for buffer in weights_buffer.values():
            candidate_state_dict = buffer.get()
            if any(torch.isnan(param).any() for param in candidate_state_dict.values()):
                logger.error(f"Candidate weights from {buffer.label} contain NaNs. Skipping update.")
                # buffer.pop()
                return
        
        if aggregation_function is federated_averaging:
            aggregated_state_dict = aggregation_function(global_model.state_dict(), [buffer.get() for buffer in weights_buffer.values()])
        else:
            aggregated_state_dict = aggregation_function(global_model.state_dict(), [buffer.get() for buffer in weights_buffer.values()], **kwargs)
        
        # Check if the aggregated state dict has no NaNs
        if any(torch.isnan(param).any() for param in aggregated_state_dict.values()):
            logger.error("Aggregated state dict contains NaNs. Skipping update.")
            return
        # pop the oldest element from each one of the buffers
        for buffer in weights_buffer.values():
            buffer.pop()
        global_model.load_state_dict(aggregated_state_dict)
        push_weights_to_vehicles()
    else:
        logger.debug(f"Waiting for more data to aggregate the weights.")


def push_weights_to_vehicles():
    weights_reporter.push_weights(global_model.state_dict())


def create_weights_buffer(vehicle_weights_topics, **kwargs):
    """
        Create a buffer for each vehicle to store the weights.
    """
    weights_buffer = {}
    for topic in vehicle_weights_topics:
        weights_buffer[topic] = GenericBuffer(size=kwargs.get('weights_buffer_size', 3), label=topic)
    return weights_buffer


def consume_weights_data(vehicle_weights_topics, **kwargs):

    consumer = create_consumer(**kwargs)

    consumer.subscribe(vehicle_weights_topics)
    logger.info(f"will start consuming {vehicle_weights_topics}")

    try:
        while not stop_threads:
            msg = consumer.poll(5.0)  
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"End of partition reached for {msg.topic()}")
                else:
                    logger.error(f"consumer error: {msg.error()}")
                continue

            deserialized_data = deserialize_message(msg)
            if deserialized_data:
                process_message(msg.topic(), deserialized_data, **kwargs)

    except KeyboardInterrupt:
        logger.info(f" Consumer interrupted by user.")
    except Exception as e:
        logger.error(f" Error in consumer: {e}")
    finally:
        consumer.close()
        logger.info(f" Consumer closed.")


def create_global_model_placeholder(**kwargs):
    return MLP(**kwargs)

def signal_handler(sig, frame):
    global stop_threads, consuming_thread, aggregation_thread
    logger.debug(f"Received signal {sig}. Gracefully stopping FL and its consumer threads.")
    stop_threads = True


def load_eval_df(kwargs):
    whole_eval_df = pd.read_csv('data/final_synth_eval_ds.csv', index_col=0)
    
    # feats
    eval_feats = whole_eval_df.drop(['cluster','class'], axis=1).values
    # replace nans with zeros and remove infs:
    eval_feats = np.nan_to_num(eval_feats, posinf=1e6, neginf=-1e6)
    # eval_feats = np.clip(eval_feats, -4000, 30000)

    # labels
    labels = whole_eval_df['class'].values
    eval_labels = [0 if label == 'Normal' else 1 for label in labels]
    
    batch_size = kwargs.get('batch_size', 32)

    # divide feats and eval labels into batches
    eval_feats = [eval_feats[i:i+batch_size] for i in range(0, len(eval_feats), batch_size)]
    eval_labels = [eval_labels[i:i+batch_size] for i in range(0, len(eval_labels), batch_size)]

    # remove last batch if it's smaller than the batch size
    if len(eval_feats[-1]) < batch_size:
        eval_feats = eval_feats[:-1]
        eval_labels = eval_labels[:-1]

    # transform to pytorch:
    eval_feats = torch.tensor(np.array(eval_feats), dtype=torch.float32)
    eval_labels = torch.tensor(np.array(eval_labels), dtype=torch.long)

    return eval_feats, eval_labels
    

def parse_str_list(arg):
    # Split the input string by commas and convert each element to int
    try:
        return [str(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Arguments must be strings separated by commas")
    

def main():
    global mode
    global logger, weights_buffer, global_model, weights_reporter, global_metrics_reporter, stop_threads
    global consuming_thread, aggregation_thread, eval_feats, eval_labels

    parser = argparse.ArgumentParser(description='Federated Learning script.')
    parser.add_argument('--logging_level', default='INFO' ,type=str, help='Logging level')
    parser.add_argument('--project_name', type=str, default="OPEN_FAIR", help='Wandb Project name')
    parser.add_argument('--run_name', type=str, default="Some run", help='Wandb run name')
    parser.add_argument('--online', action='store_true', help='Send wand metrics to the public wandb cloud')
    parser.add_argument('--kafka_broker_url', type=str, default='kafka:9092', help='Kafka broker URL')
    parser.add_argument('--kafka_consumer_group_id', type=str, default=FEDERATED_LEARNING, help='Kafka consumer group ID')
    parser.add_argument('--kafka_auto_offset_reset', type=str, default='earliest', help='Start reading messages from the beginning if no offset is present')
    parser.add_argument('--kafka_topic_update_interval_secs', type=int, default=30, help='Topic update interval for the kafka reader')
    parser.add_argument('--initialization_strategy', type=str, default="xavier", help='Initialization strategy for global model')
    parser.add_argument('--aggregation_strategy', type=str, default="fedavg", help='Aggregation strategy for FL (fedavg, fedyogi, fedmedian, fedprox)')
    parser.add_argument('--weights_buffer_size', type=int, default=3, help='Size of the buffer for weights')
    parser.add_argument('--aggregation_interval_secs', type=int, default=30, help='Aggregation interval in seconds')
    parser.add_argument('--input_dim', type=int, default=59, help='Input dimension of the model')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension of the model')
    parser.add_argument('--h_dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--layer_norm', action="store_true", help='Perform layer normalization')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--mode', type=str, default='OF', help='Mode: OF or SW')
    parser.add_argument('--probe_metrics',  type=parse_str_list, default=['RTT', 'INBOUND', 'OUTBOUND', 'CPU', 'MEM'])
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Server-side learning rate for adaptive aggregation strategies (fedyogi)')
    parser.add_argument('--yogi_beta1', type=float, default=0.9, help='FedYogi beta1 (momentum)')
    parser.add_argument('--yogi_beta2', type=float, default=0.999, help='FedYogi beta2 (adaptive)')
    parser.add_argument('--yogi_epsilon', type=float, default=1e-3, help='FedYogi epsilon for numerical stability')

    args = parser.parse_args()

    mode = args.mode
    if mode == 'SW':
        args.input_dim = args.input_dim + len(args.probe_metrics)
        args.output_dim = 4

    logging.basicConfig(format='%(name)s-%(levelname)s-%(message)s', level=str(args.logging_level).upper())
    logger = logging.getLogger('[FL]')


    # create a global model placeholder
    global_model = create_global_model_placeholder(**vars(args))
    init_global_model(**vars(args))

    # how many vehicles we have out there?
    vehicle_weights_topics = check_vehicle_topics(**vars(args))

    # create buffers for the local weights of each vehicle
    weights_buffer = create_weights_buffer(vehicle_weights_topics, **vars(args))

    # create a reporter to push the global weights to vehicles
    weights_reporter = WeightsReporter(logger=logger, **vars(args))

    # create a reporter to push the global metrics to wandb
    global_metrics_reporter = GlobalMetricsReporter(logger=logger, **vars(args))

    # load eval dataframe:
    eval_feats, eval_labels = load_eval_df(vars(args))

    logger.info(f"Starting FL with {len(vehicle_weights_topics)} in {mode} mode for vehicles: {vehicle_weights_topics}, probing metrics: {args.probe_metrics}")
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))
    stop_threads = False
    consuming_thread=threading.Thread(target=consume_weights_data, args=(vehicle_weights_topics,), kwargs=vars(args))
    consuming_thread.daemon=True
    consuming_thread.start()

    if args.aggregation_interval_secs > 0:
        # create a thread to aggregate the weights each aggregation_interval_secs:
        aggregation_thread = threading.Thread(target=aggregate_weights_periodically, kwargs=vars(args))
        aggregation_thread.start()

    while stop_threads is False:
        time.sleep(1)
    
    if args.aggregation_interval_secs > 0:
        aggregation_thread.join(1)
    consuming_thread.join(1)
    logger.info("Federated Learning stopped.")


if __name__=="__main__":
    main()
