import io
import torch
import tempfile
from fastapi import FastAPI
from app.models.dcgan import Generator
from torchvision.utils import save_image
from fastapi.responses import FileResponse

app = FastAPI()

ngpu = 0  
nz = 100  

generator_path = "nets/netG.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
