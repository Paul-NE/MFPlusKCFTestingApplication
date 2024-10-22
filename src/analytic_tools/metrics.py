class Metrics:
    def __init__(self) -> None:
        self.ious = IOUs()
        self.scale_error = {
            "sx": [],
            "sy": []
        }
        self.shift_error = {
            "dx": [],
            "dy": []
        }
    
    def update(self, ):
        pass
