from fasthtml.common import *

current_idx = -1
dir = "/Users/cosmincojocaru/keypoints_football/keypoints_football/keypoints_dataset/keypoints_dataset/train"

def get_image_paths(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_names = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not "_annotated" in file:
                continue
            if file.lower().endswith(image_extensions):
                image_names.append(file)
    
    return image_names

image_names = get_image_paths(dir)

# Flexbox CSS (http://flexboxgrid.com/)
gridlink = Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css")

# Our FastHTML app
app = FastHTML(hdrs=(picolink, gridlink))

# Main page
@app.get("/")
def get(session):
    session.setdefault('current_idx', 0)
    session['current_idx'] = session.get('current_idx') + 1
    return Div(
                       Img(src=f"/img/{image_names[session['current_idx']]}"),
    )
    
@app.get("/img/{filename}")
def get(filename: str):
    return FileResponse(f"/Users/cosmincojocaru/keypoints_football/keypoints_football/keypoints_dataset/keypoints_dataset/train/{filename}")

serve()

