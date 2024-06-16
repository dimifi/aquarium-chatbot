import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import re
import random

# Setting a seed for reproducibility
torch.random.manual_seed(0)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Text generation pipeline with the loaded model and tokenizer
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load JSON data
with open('C:/Users/dimfi/Desktop/Chat2/Apple/data.json', 'r', encoding='utf-8') as file:
    product_info = json.load(file)


# Context for product based queries
context_prompt = "You are an enthusiastic chatbot assistant for Seahorse Aquariums, designed to provide accurate and relevant information about our products. All prices are in euros. Any information asked about URLs or Models or Stock or Cabinets are given directly and concisely."

# List of follow-up questions
follow_up_questions = [
    "Do you have any other questions about our products?",
    "Can I assist you with anything else today?",
    "Is there more information you need on another product?",
    "Anything else you'd like to know about our items?"
]

# Creating context 
context = {
    "last_product": None,
    "last_category": None
}

# Creating a set of product names for fast lookup
product_names = set(product['Product'].lower() for product in product_info)

def get_product_detail(product_name, detail_type):
    # Iterate through products to find the matching product name
    for product in product_info:
        if product['Product'].lower() == product_name:
            return product.get(detail_type, "Detail not available")
    return "Product not found"

# Build a category mapping to list products by category
category_map = {}
for product in product_info:
    for category in product.get('Category', []):
        category_lower = category.lower()  
        if category_lower not in category_map:
            category_map[category_lower] = []
        category_map[category_lower].append(product['Product'])

# Now category_map['tanks'] and category_map['aquariums'] will contain lists of their respective products.
def handle_category_query(query):
    query_lower = query.lower()
    in_stock_items = []
    out_of_stock_items = []
    with_cabinet_items = []
    without_cabinet_items = []

    # Initialise the response infomration
    response_info = ""  
    for category in category_map:
        if category in query_lower:
            for product_name in category_map[category]:
                product = next((item for item in product_info if item['Product'] == product_name), None)
                if product:
                    # Filter for stock status
                    if product['Stock'].lower() == 'in stock':
                        in_stock_items.append(product['Product'])
                    else:
                        out_of_stock_items.append(product['Product'])
                    # Filter for cabinet status
                    if product.get('Cabinet Included', '').lower() == 'yes cabinet included':
                        with_cabinet_items.append(product['Product'])
                    else:
                        without_cabinet_items.append(product['Product'])

    # Determine what kind of Stock or Cabinet information the user is asking for
    if "out of stock" in query_lower or "aren't in stock" in query_lower or "are not in stock" in query_lower:
        if out_of_stock_items:
            response_info += "Out of stock: " + ", ".join(out_of_stock_items) + ". "
        else:
            response_info += "Currently, there are no items out of stock in this category. "
    elif "in stock" in query_lower or "available" in query_lower:
        if in_stock_items:
            response_info += "In stock: " + ", ".join(in_stock_items) + ". "
        else:
            response_info += "Currently, there are no items in stock in this category. "
    
    # Handling cabinet queries with explicit checks for negations
    if any(phrase in query_lower for phrase in ["without cabinet", "don't come with cabinet", "don't have cabinet", "don't include cabinet", "with no cabinet"]):
        if without_cabinet_items:
            response_info += "Without cabinet: " + ", ".join(without_cabinet_items) + ". "
        else:
            response_info += "Currently, there are no items without cabinets available. "
    elif any(phrase in query_lower for phrase in ["with cabinet", "include cabinet", "have cabinet"]):
        if with_cabinet_items:
            response_info += "With cabinet: " + ", ".join(with_cabinet_items) + ". "
        else:
            response_info += "Currently, there are no items with cabinets available. "

    if not response_info:  # General response if no specifics are found
        response_info = "Currently, there are no items in these categories."

    return response_info + "\n" + random.choice(follow_up_questions)


# Post processing
def clean_response(response):
    unwanted_phrases = [
        "as you have mentioned earlier",
        "like I said before",
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, "").strip()
    # Clean up spaces before periods, commas, and other punctuation
    response = re.sub(r'\s+([?.!,])', r'\1', response)
    return response

# Handle user query
def handle_query(query):
    query_lower = query.lower()
    global context
    
    # Check for "tank"/ "aquarium" categories / for category mentions
    if "tanks" in query_lower or "aquariums" in query_lower:
        context['last_category'] = query_lower
        category_response = handle_category_query(query_lower)
        if category_response:
            return category_response
        
    # Check for product mentions and context update
    product_mentioned = False
    for product_name in product_names:
        if product_name in query_lower:
            context['last_product'] = product_name
            product_mentioned = True
            break
        
    # Handling follow-up questions 
    if not product_mentioned and context['last_product']:
        query_lower += " " + context['last_product']
        
    # Process the query for details
    for product_name in product_names:
        if product_name in query_lower:
            detail_type = None
            if any(term in query_lower for term in ["price", "cost", "how much"]):
                detail_type = "Price"
            elif any(term in query_lower for term in ["specifications", "specs", "details"]):
                detail_type = "Specifications"
                product_detail = get_product_detail(product_name, detail_type)
                # Handle specifications separately
                return f"Here are the specifications for the {product_name.title()}:\n{product_detail}\n{random.choice(follow_up_questions)}"
            elif any(term in query_lower for term in ["url", "link", "product page", "webpage"]):
                detail_type = "URL"
            elif any(term in query_lower for term in ["model", "product number"]):
                detail_type = "Model"
            elif any(term in query_lower for term in ["stock", "available", "availability"]):
                detail_type = "Stock"
            elif any(term in query_lower for term in ["cabinet", "include cabinet", "come with cabinet", "have cabinet"]):
                detail_type = "Cabinet Included"

            if detail_type:
                product_detail = get_product_detail(product_name, detail_type)
                # Response prompt
                response_prompt = f"{context_prompt} For the {product_name.title()}, {product_detail}. Now, {query}\nA:"
            else:
                response_prompt = f"{context_prompt} Tell me more about {product_name.title()} {query}\nA:"

            # Generate the response with the Configured settings
            response = pipe(response_prompt, max_new_tokens=100, do_sample=True, temperature=0.2, top_p=0.95, repetition_penalty=1.2)[0]['generated_text']  
          
            # Extract and clean the response after 'A:'
            extracted_response = response.split("\nA:")[1].split("\n")[0] if "\nA:" in response else response
            cleaned_response = clean_response(extracted_response)
            return cleaned_response.strip() + "\n" + random.choice(follow_up_questions)
    
    # Context for general queries    
    context_prompt2 = "You are a chatbot assistant for the store Seahorse Aquariums trained to provide concise and relevant answers to the user's question."
    # Combine the context and the query into a single prompt for the model
    response_prompt = f"{context_prompt2} {query}\nA:"
    response = pipe(response_prompt, max_length=150, do_sample=True, temperature=0.3, top_p=0.95, top_k=20, repetition_penalty=1.1)[0]['generated_text']

    # Post-process the response to remove any content after an initial reply
    # Split the response at "A:" and take the first segment
    # Also, trim any additional dialogue that might follow after a line break or user prompt
    cleaned_response = response.split("\nA:")[1].split("\n")[0] if "\nA:" in response else response
    return cleaned_response.strip()


# User input
print(handle_query("hey there! How are you?"))


# C h a t b o t :)
import tkinter as tk

# Initialize main application window
root = tk.Tk()
root.title("Chatbot") # Title of the window

# Set up the chat history text area
text_area = tk.Text(root, height=20, width=50)
text_area.pack(pady=10)

# Input field for the user's message
entry_field = tk.Entry(root, width=40)
entry_field.pack(pady=10)

def handle_query_gui(event=None):
    
    """
    Handle the user's query from the GUI, generate a response, and update the chat history.
    
    Args:
    event: The event that triggered this function (e.g., pressing the "Enter" key).
    """
    user_query = entry_field.get()
    entry_field.delete(0, tk.END)  # Clear the input field
    # Display user's query in the text area
    text_area.insert(tk.END, "You: " + user_query + "\n")
    # Generate response using the model
    response = handle_query(user_query)
    text_area.insert(tk.END, "Bot: " + response + "\n\n")

# Button to send the message
send_button = tk.Button(root, text="Send", command=handle_query_gui)
send_button.pack()

# Bind the "Enter" key to send replies
entry_field.bind('<Return>', handle_query_gui)

# Ensure the entry field is focused
entry_field.focus_set()

# Run the application
root.mainloop()