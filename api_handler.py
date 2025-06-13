import torch
import time

# Define a simple API class to interact with a machine learning model
class API():
    def __init__(self):
        # Track the number of requests made to the handler
        self.num_req = 0

    def handler(self, model, data, rate_limit=False, 
                rounding=False, hard_label=False):
        self.num_req += 1 # Increment the request counter

        if rate_limit == True: # If rate limiting is enabled, pause after 20 requests
            if self.num_req == 20:
                print("You reach the query limit. Please wait for 10 seconds")
                time.sleep(10)  # Simulate API cooldown
                self.num_req = 0  # Reset counter after cooldown
        
        # If rounding is enabled, round model output to 1 decimal place
        if rounding == True: 
            return torch.round(model(data), decimals=1)

        # If hard label is enabled, return class with highest confidence
        if hard_label == True: 
            return model(data).argmax(dim=1)

        return model(data)
