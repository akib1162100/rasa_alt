import re
number_string = re.findall(r'\d{10}',"my account 12341243 number is 1234567890")
print(number_string)