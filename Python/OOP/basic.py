class Employee():
    def __init__(self, name, title, PayPerhour = None) -> None:
        self.name = name
        self.title = title
        if PayPerhour is not None:
            PayPerhour = float(PayPerhour)
        self.PayPerhour = PayPerhour
    def getName(self):
        return self.name
    def getTitle(self):
        return self.title
    def getPayPerhour(self):
        return self.PayPerhour
    