"""
This module contains a few render functions used in the project, namely:
- general render function (printmd)
- display type options (display_folder_version)
- display content of selected folder (display_initial_folder_content)
- display content of target folder (display_final_folder_content).
"""


from IPython.display import Markdown, display, HTML
import os
import pprint
cwd = os.getcwd()
css_url = os.path.join(cwd,"Css","script.css")


# ------------------------------------------------------------------------------------------------------------  
# ------------------------------------------------------------------------------------------------------------

def load_css():
    with open(css_url) as style_file:
        style = style_file.read()
    display(HTML(style))
    return

    
# ------------------------------------------------------------------------------------------------------------  
# ------------------------------------------------------------------------------------------------------------ 


def printmd(string):
    """ Display corresponding  html code."""
    display(Markdown(string))
    return

    
def script_title(string,thickness="ultrathick"):
    """ displays section title """
    printmd(f"<div style='font-size:400%;font-weight:900;line-height:125%'><br>{string}</div>")
    printmd(f"<div class='separatrix {thickness}'></div>")  
    return


def title(string,thickness="verythick"):
    """ displays section title """
    printmd(f"<span style='font-size:240%;font-weight:900'><br>{string}</span>")
    printmd(f"<div class='separatrix {thickness}'></div>")  
    return

def separatrix(thickness="verythick"):
    printmd(f"<div class='separatrix {thickness}'></div>")    
    return


def start():
    title("Script Start")
    separatrix()
    return


def end():
    title("Script End")
    separatrix()    
    return
 

def legend(string,thickness="verythick"):
    new_string = f"<br><div style='font-size:240%;font-weight:900;width:20%;float:left;height:150px;padding:10px;'>Legend</div>"
    new_string += f"<div style='width:80%;float:left;border-left:10px solid #4682B4;padding:0;padding-left:50px;height:120px;'>\
                    {string}</div>"
    printmd(new_string)  
    printmd(f"<br><div class='separatrix {thickness}'></div><br>")  
    return

# ------------------------------------------------------------------------------------------------------------  
# ------------------------------------------------------------------------------------------------------------ 
    

 