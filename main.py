from random import randint
from pandas import json_normalize
from typing import Set, Any
from fastapi import FastAPI
from kafka import TopicPartition
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiokafka import AIOKafkaProducer

import matplotlib.pyplot as plt
import pickle
import time
import pyspark

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import *

import pandas as pd
import uvicorn
import aiokafka
import asyncio
import json
import logging
import os

# instantiate the API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ourPred = []
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()

model = DecisionTreeClassificationModel.load("dtc_model")
# global variables
arguments = None
resultat_pred = None
consumer_task = None
consumer = None
_state = 0
df = pd.read_csv("diabetes.csv")
to_describe = pd.DataFrame(
    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
             'Age', 'Outcome'])
# env variables
KAFKA_TOPIC = "diabetes"
KAFKA_CONSUMER_GROUP_PREFIX = os.getenv('KAFKA_CONSUMER_GROUP_PREFIX', 'group')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

# initialize logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    log.info('Initializing API ...')
    await initialize()
    await consume()


@app.on_event("shutdown")
async def shutdown_event():
    log.info('Shutting down API')
    consumer_task.cancel()
    await consumer.stop()


@app.get("/loadcsv")
def load_csv():
    async def iter_df():
        for _, res in df.iterrows():
            await asyncio.sleep(2)
            row1 = json.dumps(res.to_dict())
            row = json.loads(row1)
            tp = [int(row['Pregnancies']), int(row['Glucose']), int(row['BloodPressure']), int(row['SkinThickness']),
                  int(row['Insulin']), int(row['BMI']), int(row['DiabetesPedigreeFunction']), int(row['Age']),
                  int(row['Outcome'])]
            described = await csvproduce(tp)
            yield described
    return StreamingResponse(iter_df(), media_type="application/json")


class patient(BaseModel):
    pregnancies: int
    glucose: int
    bloodPressure: int
    skinThickness: int
    insulin: int
    bMI: int
    diabetesPedigreeFunction: int
    age: int


@app.post("/predict/")
async def create_item(p: patient):
    await produce(p)
    return p


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/state")
async def state():
    return {"state": _state}


async def initialize():
    loop = asyncio.get_event_loop()
    global consumer
    group_id = f'{KAFKA_CONSUMER_GROUP_PREFIX}-{randint(0, 10000)}'
    log.debug(f'Initializing KafkaConsumer for topic {KAFKA_TOPIC}, group_id {group_id}'
              f' and using bootstrap servers {KAFKA_BOOTSTRAP_SERVERS}')
    consumer = aiokafka.AIOKafkaConsumer(KAFKA_TOPIC, loop=loop,
                                         bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                                         group_id=group_id)
    # get cluster layout and join group
    await consumer.start()

    partitions: Set[TopicPartition] = consumer.assignment()
    nr_partitions = len(partitions)
    if nr_partitions != 1:
        log.warning(f'Found {nr_partitions} partitions for topic {KAFKA_TOPIC}. Expecting '
                    f'only one, remaining partitions will be ignored!')
    for tp in partitions:

        # get the log_end_offset
        end_offset_dict = await consumer.end_offsets([tp])
        end_offset = end_offset_dict[tp]

        if end_offset == 0:
            log.warning(f'Topic ({KAFKA_TOPIC}) has no messages (log_end_offset: '
                        f'{end_offset}), skipping initialization ...')
            return

        log.debug(f'Found log_end_offset: {end_offset} seeking to {end_offset - 1}')
        consumer.seek(tp, end_offset - 1)
        msg = await consumer.getone()
        log.info(f'Initializing API with data from msg: {msg}')

        # update the API state
        # _update_state(msg)
        return


@app.get("/consume")
async def consume():
    global consumer_task
    global arguments
    consumer_task = asyncio.create_task(send_consumer_message(consumer))
    # await send_consumer_message(consumer)
    return {"résultat": resultat_pred}


@app.get("/stopconsumer")
async def stopconsumer():
    consumer.stop()


async def send_consumer_message(consumer):
    loop = asyncio.get_event_loop()
    global arguments
    global resultat_pred
    # get cluster layout and join group KAFKA_CONSUMER_GROUP	
    try:
        # consume messages
        async for msg in consumer:
            # x = json.loads(msg.value)
            log.info(f"Consumed msg: {msg}")
            # arguments = json.dumps(msg.value.decode())
            arguments = json.loads(msg.value.decode())
            To_Predict = Vectors.dense(arguments)
            val = model.predict(To_Predict)
            print("##################################################################")
            print(To_Predict)
            if val == 1.0:
                resultat_pred = "tested positive"
            else:
                resultat_pred = "tested negative"
            ourPred = []
            # update the API state
    #            _update_state(msg)
    finally:
        # will leave consumer group; perform autocommit if enabled
        log.warning('Stopping consumer')
        # await consumer.stop()
    loop.run_until_complete(consume())


def _update_state(message: Any) -> None:
    value = json.loads(message.value)
    global _state
    _state = value['state']


async def produce(p: patient):
    loop = asyncio.get_event_loop()
    producer = AIOKafkaProducer(loop=loop, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    # get cluster layout and initial topic/partition leadership information
    await producer.start()
    try:
        ourPred = [p.pregnancies, p.glucose, p.bloodPressure, p.skinThickness, p.insulin, p.bMI,
                   p.diabetesPedigreeFunction, p.age]
        value_json = json.dumps(ourPred).encode('utf-8')
        await producer.send_and_wait(KAFKA_TOPIC, value_json)
        print("{} Produced  ".format(time.time()))
        ourPred = []
        time.sleep(2)
    finally:
        # wait for all pending messages to be delivered or expire.
        await producer.stop()


async def csvproduce(row: []):
    loop = asyncio.get_event_loop()
    producer = AIOKafkaProducer(loop=loop, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    # get cluster layout and initial topic/partition leadership information
    await producer.start()
    try:
        to_describe.loc[len(to_describe)] = row
        Describe = {"Means": {"Pregnancies_Mean": to_describe["Pregnancies"].mean(),
                              "Glucose_Mean": to_describe["Glucose"].mean(),
                              "BloodPressure_Mean": to_describe["BloodPressure"].mean(),
                              "SkinThickness_Mean": to_describe["SkinThickness"].mean(),
                              "Insulin_Mean": to_describe["Insulin"].mean(),
                              "BMI_Mean": to_describe["BMI"].mean(),
                              "DiabetesPedigreeFunction_Mean": to_describe["DiabetesPedigreeFunction"].mean(),
                              "Age_Mean": to_describe["Age"].mean(),
                              "Outcome_Mean": to_describe["Outcome"].mean()},
                    "Maxs": {"Pregnancies_Max": int(to_describe["Pregnancies"].max()),
                             "Glucose_Max": int(to_describe["Glucose"].max()),
                             "BloodPressure_Max": int(to_describe["BloodPressure"].max()),
                             "SkinThickness_Max": int(to_describe["SkinThickness"].max()),
                             "Insulin_Max": int(to_describe["Insulin"].max()),
                             "BMI_Max": int(to_describe["BMI"].max()),
                             "DiabetesPedigreeFunction_Max": int(to_describe["DiabetesPedigreeFunction"].max()),
                             "Age_Max": int(to_describe["Age"].max()),
                             "Outcome_Max": int(to_describe["Outcome"].max())},
                    "Mins": {"Pregnancies_Min": int(to_describe["Pregnancies"].min()),
                             "Glucose_Min": int(to_describe["Glucose"].min()),
                             "BloodPressure_Min": int(to_describe["BloodPressure"].min()),
                             "SkinThickness_Min": int(to_describe["SkinThickness"].min()),
                             "Insulin_Min": int(to_describe["Insulin"].min()),
                             "BMI_Min": int(to_describe["BMI"].min()),
                             "DiabetesPedigreeFunction_Min": int(to_describe["DiabetesPedigreeFunction"].min()),
                             "Age_Min": int(to_describe["Age"].min()),
                             "Outcome_Min": int(to_describe["Outcome"].min())},
                    "Counts": {"Pregnancies_Count": int(to_describe["Pregnancies"].count()),
                               "Glucose_Count": int(to_describe["Glucose"].count()),
                               "BloodPressure_Count": int(to_describe["BloodPressure"].count()),
                               "SkinThickness_Count": int(to_describe["SkinThickness"].count()),
                               "Insulin_Count": int(to_describe["Insulin"].count()),
                               "BMI_Count": int(to_describe["BMI"].count()),
                               "DiabetesPedigreeFunction_Count": int(to_describe["DiabetesPedigreeFunction"].count()),
                               "Age_Count": int(to_describe["Age"].count()),
                               "Outcome_Count": int(to_describe["Outcome"].count())}}
        # yield json.dumps(Describe).encode('utf-8') + '\n'
        value_json = json.dumps(Describe,indent=4).encode('utf-8')
        print(value_json)
        await producer.send_and_wait(KAFKA_TOPIC, value_json)
        print("{} Produced  ".format(time.time()))
        time.sleep(3)
        return value_json
    finally:
        # wait for all pending messages to be delivered or expire.
        await producer.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
