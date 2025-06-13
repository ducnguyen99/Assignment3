import torch
import time

class API():
    def __init__(self):
        # Initialize the API instance with a counter set to 0
        self.num_req = 0

    def handler(self, model, data, rate_limit=False, rounding=False, hard_label=False):
        # Increment the rate limit counter with each request
        self.num_req += 1

        # Check if the rate limit has reached 100 requests
        if rate_limit == True:
          if self.num_req == 20:
              print("You reach the query limit. Please wait for 10 seconds")
              time.sleep(10)
              self.num_req = 0

        if rounding == True:
            return torch.round(model(data), decimals=1)

        if hard_label == True:
          return model(data).argmax(dim=1)

        # Return the rounded output
        return model(data) 
