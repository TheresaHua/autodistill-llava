from autodistill_llava import LLaVA
from autodistill.detection import CaptionOntology

ontology=CaptionOntology({
    "milk bottle": "bottle",
    "blue cap": "cap"
})

base_model = LLaVA(ontology)
detectionDataset = base_model.label("./image/milk-video-8-00039.png", extension="_test")
print(detectionDataset)


from autodistill_llava import LLaVA
from autodistill.detection import CaptionOntology

ontology=CaptionOntology({"milk bottle": "bottle","blue cap": "cap"})
model = LLaVA(ontology=ontology)
resault = model.predict("/home/k100/autodistill/data/temp_images/milk-video-8-00039.png")
