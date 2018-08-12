
file = open('alice_in_wonderland_raw.txt', 'r')
result = ''
for line in file:
    if line is not '\n':
        stripped = line.strip()
        txt = stripped.replace('  ', ' ')
        txt = txt.replace('_', '')
        txt = txt.replace('`', "")
        txt = txt.replace("'", "")
        txt = txt.replace('\\', "")
        txt = txt.replace('--', "")
        txt = txt.replace('/', "")
        txt = txt.lower()
        result += txt
        result += ' '
        
file = open('alice_in_wonderland.txt', 'w')
file.write(result)
print(''.join(set(result)))
