# Chatbot Assistant for Product Information Retrieval

## Overview
This project is a chatbot assistant designed to provide accurate and relevant information about aquarium products using NLP techniques, Python, and TensorFlow.

## Features
- Dynamic information retrieval from a JSON file
- Contextual prompting and text processing
- Interactive follow-up questions

## Technologies Used
- Python
- TensorFlow
- Scrapy
- Selenium
- JSON

## System Architecture
+--------------+
|  User Input  |
+--------------+
       |
       v
+--------------+
| Preprocessing|
+--------------+
       |
       v
+--------------+
|Data Retrieval|
+--------------+
       |
       v
+------------------+
| Model Processing |
+------------------+
       |
       v
+---------------+
| Postprocessing|
+---------------+
       |
       v
+--------------+
|   Output     |
+--------------+


## Setup
- The used model is the Phi3-mini
- Make sure you installed the required libraries
- The chatbot script is called aquarium_chatbot_assistant: https://github.com/dimifi/aquarium-chatbot/blob/main/aquarium_chatbot_assistant.py
- The data file the chatbot uses to retrieve data is called product_listings_data: https://github.com/dimifi/aquarium-chatbot/blob/main/product_listings_data.json


## Challenges
- Model selection with limited hardware resources
- Crafting effective context prompts
- Handling dynamic web content and pop-up ads during data crawling


## Areas for Improvement
- Implementing synonym recognition for product names
- Enhancing context management
- Optimizing response times
