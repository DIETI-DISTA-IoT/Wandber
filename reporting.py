from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
import pickle

class WeightsReporter:
    def __init__(self, logger, **kwargs):
        conf_prod_weights={
        'bootstrap.servers': kwargs.get('kafka_broker_url'),  # Kafka broker URL
        'key.serializer': StringSerializer('utf_8'),
        'value.serializer': lambda v, ctx: pickle.dumps(v)
         }
        self.producer = SerializingProducer(conf_prod_weights)
        self.logger = logger

    def push_weights(self, weights):
        weights_topic=f"global_weights"
        try:
            self.producer.produce(topic=weights_topic, value=weights)
            self.producer.flush()
            self.logger.info(f"Sent global weights to topic: {weights_topic}")
        except Exception as e:
            self.logger.error(f"Failed to send global weights: {e}")
