
import re
number_string = re.findall(r'\d+',"my account 12341243 number is 1234567890")
print(number_string)


sentence= "আমার ডিউ কত কার্ড এর আউটস্টেন্ডং কত আমার বিল কত"
index = sentence.find("আউটস্টেন্ডং")
print(index)


values = ["ডিউ", "আউটস্টেন্ডং", "বিল"]

pattern = re.compile("|".join(r"\b" + re.escape(x) + r"\b" for x in values))
temp = []

temp.extend([x.upper() for x in pattern.findall(sentence, re.IGNORECASE)])
print(temp) 