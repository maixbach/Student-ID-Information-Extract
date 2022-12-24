import craft
import ocr

path = ''
craft = craft.craft(path)
res = ocr.ocr(craft)
print(res)