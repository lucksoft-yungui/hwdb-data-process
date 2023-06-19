import struct

tag_code = 11776  # The tag code you got from GNT file
tagcode_unicode = struct.pack('>H', tag_code).decode('gb2312')  # Decode GB2312 to Unicode
print(tagcode_unicode) 