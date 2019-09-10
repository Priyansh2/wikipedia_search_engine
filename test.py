text="""{{Infobox ice hockey player
| name = Marko Virtanen
| image =
| caption =
| image_size = 225px
| played_for = '''''[[SM-liiga]]'''''&lt;br&gt;[[JYP Jyväskylä]]&lt;br&gt;'''''[[Swedish Hockey League|Elitserien]]'''''&lt;br&gt;[[Färjestad BK]]&lt;br&gt;[[Södertälje SK]]
| position = [[Winger (ice hockey)|Right Wing]]
| height_ft = 5
| height_in = 10
| weight_lb = 181
| shoots = Left
| birth_date = {{birth date and age|1968|12|10}}
| birth_place = [[Jyväskylä]], [[Finland]]
| draft = Undrafted
| career_start = 1988
| career_end = 2000
}}"""
def extract_infobox(text):
	import re
	infobox=[]
	for match in reversed(list(re.finditer("{{infobox", text))):
		start=match.span()[0]
		end=start+2
		flag=2
		content=""
		for ch in text[start+2:]:
			end+=1
			if flag==0:
				break
			if ch=="{":
				flag+=1
			elif ch=="}":
				flag+=- 1
			else:
				content+=ch
		text=text[:start]+text[end:]
		infobox.append(content)
	infobox = " ".join(infobox)
	infobox_info=''

	for line in infobox.split("|"):
		#print(line)
		try:
			infobox_info+=line.split("=")[1]+" "
		except:
			infobox_info+=line+" "
	print(infobox_info)
	'''infobox_info = filter_contents(infobox_info)
	return infobox_info,text'''
#print(text)
extract_infobox(text.lower())