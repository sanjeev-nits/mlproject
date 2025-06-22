import sys
from src.logger import logging

def error_massage_details(error, error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    error_massage=f"Error occured python script name [{exc_tb.tb_frame.f_code.co_filename}] line number [{exc_tb.tb_lineno}] error message [{ str(error)}]"
    return error_massage


class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_massage_details(error_message, error_details=error_details)
    
    def __str__(self):
        return self.error_message
    
