from flask import Flask, request
import os
import json
import pandas as pd
import random
import torch
import time


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/webhook",methods = ["POST"])
def test():
    data = request.get_json(silent = True,force=True)
    json_data = json.dumps(data)
    print(type(data))
    data_dict = json.loads(json_data)
    display_name = data_dict['queryResult']['intent']['displayName']
    if display_name == 'Story Telling - yes':
        # display_name = "I'm excited to share a story with you! Let's pick one together. Do you have any specifications in the story?"
        fulfillmentText = display_name
    # search_value = 'target_value'
        df = pd.read_csv('your_dataframe.csv')
        print(df.columns) 
        unique_cities = df['cities'].unique().tolist()
        unique_cities = unique_cities[:5]
        unique_names = df['personas'].unique().tolist()
        unique_names = unique_names[:5]
        unique_genres = df['predicted_genre'].unique().tolist()
        unique_genres = unique_genres[:5]
        # Convert each element to string before joining
        display_name = f"""
            Do you have any specifications in the story? Select Genre Example: {', '.join(str(genre) for genre in unique_genres)} 
            Enter character name: {', '.join(str(name) for name in unique_names)} 
            Enter a city if you want: {', '.join(str(city) for city in unique_cities)}"""
    elif display_name == 'Story Telling - yes - more':
        genre = data_dict['queryResult']['parameters']['Genre']
        city = data_dict['queryResult']['parameters']['location']['city']
        print(genre, city)
        personas = [person["name"] for person in data_dict["queryResult"]["parameters"]["person"]]

        variations = [
        f"Imagine a {genre} story set in {city}, featuring characters like {', '.join(personas)}. What adventures await them?",
        f"Write a tale from the {genre} genre involving {' and '.join(personas)} in the city of {city}. How do they overcome the challenges they face?",
        f"In the {genre} world of {city}, {' and '.join(personas)} encounter a mysterious challenge. Craft their story."
        ]
        display_name = random.choice(variations)
        # display_name = generate_story(display_name, 200)
        display_name = """Once upon a time, in the bustling city of New York, there lived a curious little boy named Alex. Alex had an insatiable thirst for knowledge and a deep passion for learning about different countries and their cultures.
            One sunny afternoon, while immersed in a book in his cozy library, Alex stumbled upon a peculiar sentence that left him feeling uneasy: "Alexandria is a big, beautiful country with a long history."
            Perplexed and intrigued, Alex sought guidance from his wise teacher, Mr. Harper, a brilliant linguist known for his profound knowledge.
            "Mr. Harper," asked Alex with furrowed brows, "why does this sentence make me feel uncomfortable?"
            With a gentle smile, Mr. Harper explained the concept of subject-verb agreement to Alex. He likened it to the training wheels on a bicycle, aiding in the comprehension of intentions and actions.
            "Just as we agree to meet at a specific time or place," Mr. Harper elucidated, "subject-verb agreement assists us in understanding the purpose behind someone's words."
            Patiently, Mr. Harper illustrated how words morph in meaning based on their usage. For instance, when conveying emotions, like "Alexandria," one could interpret it as "He feels sad" or "He desires exploration of new realms."
            Enlightened by Mr. Harper's wisdom, Alex's curiosity soared to greater heights as he embarked on a journey of linguistic exploration, eager to unravel the mysteries of language and communication."""

        
    fulfillmentText = display_name
    return {'fulfillmentText':fulfillmentText}

def generate_story(prompt, max_length=200):

    # Directory where your model is saved
    output_dir = "fine_tuned_gpt2_model_10epochs_500_texts_200_words"
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    output = model.generate(input_ids, max_length=max_length, temperature=0.7, num_return_sequences=1)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story
    # else:
    #     # This part is executed if the loop completes without breaking, indicating value not found
    #     print(f"Value {search_value} not found in the nested dictionary.")
   


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))

    print ("Starting app on port %d" %(port))

    app.run(debug=True, port=port, host='0.0.0.0')