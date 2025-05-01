# This file does not contain any langchain code.
# Codes present in this file are written simply to understand "How Runnables work in
# the background? "


from abc import ABC, abstractmethod
import random


class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass


class FakeLLM(Runnable):

    def __init__(self):
        super().__init__()
        print("LLM Created.")

    def invoke(self, prompt):
        response_list = [
            "Langchain was created by Harrison Chase",
            "Delhi is the Capital of India.",
            "The hardest bone in the human body is the femur.",
        ]

        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            "Langchain was created by Harrison Chase",
            "Delhi is the Capital of India.",
            "The hardest bone in the human body is the femur.",
        ]

        return {"response": random.choice(response_list)}


class FakePromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        super().__init__()
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


class FakeStrOutputParser(Runnable):
    def __init__(self):
        super().__init__()

    def invoke(self, input_dict):
        return input_dict["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        super().__init__()
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            ### The following is a very important line
            # Here we are sending the output of one Runnable to  
            # the input of another Runnable using invoke() method.
            input_data = runnable.invoke(input_data)

        return input_data


