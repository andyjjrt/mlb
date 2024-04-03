from model import MLBPredictor

if __name__ == "__main__":
    model = MLBPredictor()
    model.train(start_year=1900)
