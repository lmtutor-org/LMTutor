import os,glob
import re
import fitz
from PIL import Image 
from pytesseract import pytesseract
from pdf2image import convert_from_path


#file_path: folder path for book pdf  
#file_path_to_save: folder path for all ouput txt file
def pdf_to_txt_book(file_path,file_path_to_save):
    for filename in glob.glob(os.path.join(file_path, '*.pdf')):
        file_name_txt=file_path_to_save+ "/"+ re.findall(r'\/([^\/]+)\.pdf',filename)[0]+".txt"
        with open(file_name_txt, 'w') as file:
            text=''
            doc_opened=fitz.open(filename)
            for page in doc_opened:
                text += page.get_text()
            thrshold=len(re.findall('\n', text))/len(text)
            text_to_write=''
            for i in range(len(doc_opened)):
                page=doc_opened[i]
                output = page.get_text("blocks")
                for j in range(len(output)):
                    block_text=output[j][4]
                    if block_text.startswith('Table') or block_text.startswith('Figure'):
                        continue
                    elif len(re.findall(r'\.',block_text))>30:
                        continue
                    elif (len(re.findall('\n', output[j][4]))/len(output[j][4])<=thrshold) and (output[j][6] == 0) :
                        temp= block_text.replace('-\n','')
                        text_to_write += temp.replace('\n', ' ')+ "\n"
                    else:
                        continue
            file.write(text_to_write)
            
#file_path: folder path for paper pdf  
#file_path_to_save: folder path for all ouput txt file            
def pdf_to_txt_paper(file_path,file_path_to_save):
    for filename in glob.glob(os.path.join(file_path, '*.pdf')):
        file_name_txt=file_path_to_save+ "/"+ re.findall(r'\/([^\/]+)\.pdf',filename)[0]+".txt"
        with open(file_name_txt, 'w') as file:
            text=''
            doc_opened=fitz.open(filename)
            for page in doc_opened:
                text += page.get_text()
            thrshold=len(re.findall('\n', text))/len(text)
            text_to_write=''
            for i in range(len(doc_opened)):
                page=doc_opened[i]
                output = page.get_text("blocks")
                for j in range(len(output)):
                    block_text=output[j][4]
                    if block_text.startswith('Table') or block_text.startswith('Figure'):
                        continue
                    elif re.match(r'^[1-9][a-f]+', block_text):
                        continue
                    elif (len(re.findall('\n', output[j][4]))/len(output[j][4])<=thrshold) and (output[j][6] == 0) :
                        temp= block_text.replace('-\n','')
                        text_to_write += temp.replace('\n', ' ')+ "\n"
                    else:
                        continue
            file.write(text_to_write)
            
            
#file_path: folder path for slides pdf  
#file_path_to_save: folder path for all ouput txt file
def pdf_to_txt_slides(file_path,file_path_to_save):
    for filename in glob.glob(os.path.join(file_path, '*.pdf')):
        file_name_txt=file_path_to_save+ "/"+ re.findall(r'\/([^\/]+)\.pdf',filename)[0]+".txt"
#         print(file_name_txt)
        with open(file_name_txt, 'w') as file:
            doc_opened=fitz.open(filename)
            text_to_write=''
            for i in range(len(doc_opened)):
                page=doc_opened[i]
                output = page.get_text("blocks")
                for j in range(len(output)):
                    block_text=output[j][4]
                    if len(re.findall(r'[^\w\s]', block_text))/len(re.sub(r'\s', '', block_text)) >=0.5:
                        continue
                    elif output[j][6] == 1:
                        continue
                    elif j == (len(output)-1):
                        text_to_write+=block_text+". "
                    else:
                        text_to_write+=block_text
            file.write(text_to_write)    

#file_path: folder path for latex pdf 
#file_path_img: folder path for changed image from latex pdf
#file_path_to_save: folder path for all ouput txt file
def pdf_to_txt_latex(file_path, file_path_img, file_path_to_save):
    for filename in glob.glob(os.path.join(file_path, '*.pdf')):
        file_name_txt=file_path_to_save+ "/"+ re.findall(r'\/([^\/]+)\.pdf',filename)[0]+".txt"
#         file_name_img=file_path_img+"/"+re.findall(r'\/([^\/]+)\.pdf',filename)[0]+' '+str(count)+'.jpeg'
#         print(filename)
        images = convert_from_path(filename)
        text_to_write=''
        for i in range(len(images)):
            file_name_img=file_path_img+'/'+ re.findall(r'\/([^\/]+)\.pdf',filename)[0]+ '-page'+ str(i+1)+'.png'
            images[i].save(file_name_img, 'PNG')
            image=Image.open(file_name_img)
            text=pytesseract.image_to_string(image)
            pattern = r'\d{1,2}\/\d{1,2}\/\d{2}, \d{1,2}:\d{2}'
            temp= re.sub(pattern, '', text)
            text_to_write+=temp
        with open(file_name_txt, 'w') as file:
            file.write(text_to_write)  
            file.close()
            
            
            
# change all pdf into txt file and stored in one folder DSC-291-temp            
# pdf_to_txt_book("/Users/lichenghu/desktop/DSC-291-book","/Users/lichenghu/desktop/DSC-291-temp")
# pdf_to_txt_paper("/Users/lichenghu/desktop/DSC-291-paper","/Users/lichenghu/desktop/DSC-291-temp")
# pdf_to_txt_slides("/Users/lichenghu/desktop/DSC-291-lecture","/Users/lichenghu/desktop/DSC-291-temp")
# pdf_to_txt_latex("/Users/lichenghu/desktop/DSC-291-latex","/Users/lichenghu/desktop/DSC-291-img","/Users/lichenghu/desktop/DSC-291-temp")

