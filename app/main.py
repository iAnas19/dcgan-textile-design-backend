import io
import cv2
import numpy as np
import torch
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.models.dcgan import Generator  
from torchvision.utils import save_image

from app.models import RRDBNet_arch as arch

app = FastAPI()

ngpu = 0  
nz = 100  

generator_path = "nets/netG.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'nets/RRDB_ESRGAN_x4.pth' 
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(generator_path, map_location=device))
netG.eval()

@app.get("/")
def read_root():
    return {"message": "Server up and running!"}

@app.get("/generate")
async def generate_image():
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        generated_image = netG(noise).detach().cpu()
        generated_image = (generated_image + 1) / 2  # Rescale to [0, 1]

        # Save the generated image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
            save_image(generated_image, temp_img.name, format="PNG")

        # Close the temporary file
        temp_img.close()

        # Return the temporary file using FileResponse
        return FileResponse(temp_img.name, media_type="image/png")

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_img:
        contents = await file.read()
        temp_img.write(contents)

    # Load the saved image using OpenCV
    img = cv2.imread(temp_img.name, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Enhance the image using the ESRGAN model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # Create a unique filename for the enhanced image
    filename = f"{uuid.uuid4().hex}_enhanced.png"

    # Save the enhanced image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=filename) as temp_enhanced_img:
        cv2.imwrite(temp_enhanced_img.name, output)

    # Close the temporary files
    temp_img.close()
    temp_enhanced_img.close()

    # Return the enhanced image using FileResponse
    enhanced_response = FileResponse(
        temp_enhanced_img.name,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )

    return enhanced_response
