from re import sub
textToConvert = input()

def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].lower(), s[1:]])

print( camel_case(textToConvert))
