import csv

from xenopy import Query
from os.path import exists

birds = ["Columba oenas", "Columba palumbus", "Corvus corax", "Hooded crow", "Carrion crow", "Western Jackdaw",
         "Coturnix coturnix", "Cuculus canorus", "Cyanistes caeruleus", "Dendrocopos major",
         "Middle spotted woodpecker", "Lesser spotted woodpecker", "Dryocopus martius", "Emberiza calandra",
         "Emberiza citrinella", "Erithacus rubecula", "Falco tinnunculus", "Pica pica", "Accipiter gentilis",
         "Acrocephalus palustris", "Acrocephalus scirpaceus", "Aegithalos caudatus", "Alauda arvensis", "Anser anser",
         "Anthus pratensis", "Anthus trivialis", "Buteo buteo", "Common linnet", "Carduelis carduelis",
         "European Greenfinch", "Eurasian Siskin", "Certhia brachydactyla", "Certhia familiaris", "Ciconia ciconia",
         "Coccothraustes coccothraustes", "Oenanthe oenanthe", "Oriolus oriolus", "European Crested Tit", "Parus major",
         "Willow tit", "Marsh tit", "Passer domesticus", "Passer montanus", "Periparus ater", "Pernis apivorus",
         "Phasianus colchicus", "Phoenicurus ochruros", "Phoenicurus phoenicurus", "Phylloscopus bonelli",
         "Phylloscopus collybita", "Phylloscopus sibilatrix", "Phylloscopus trochilus", "Ficedula hypoleuca",
         "Ficedula parva", "Fringilla coelebs", "Fringilla montifringilla", "Gallinago gallinago",
         "Garrulus glandarius", "Grus grus", "Hippolais icterina", "Jynx torquilla", "Lanius collurio",
         "Locustella naevia", "Loxia curvirostra", "Lullula arborea", "Luscinia megarhynchos", "Milvus migrans",
         "Milvus milvus", "Motacilla alba", "Motacilla flava", "Muscicapa striata", "Picus viridis",
         "Prunella modularis", "Pyrrhula pyrrhula", "Common Firecrest", "Regulus regulus", "Saxicola rubetra",
         "Saxicola rubicola", "Scolopax rusticola", "Sitta europaea", "Streptopelia turtur", "Strix aluco",
         "Sturnus vulgaris", "Sylvia atricapilla", "Sylvia borin", "Sylvia communis", "Lesser whitethroat",
         "Troglodytes troglodytes", "Turdus iliacus", "Turdus merula", "Turdus philomelos", "Picus canus",
         "Turdus pilaris", "Turdus viscivorus", "Vanellus vanellus"]

q = Query(name=birds[0], len_lt=60)

q.retrieve_recordings(multiprocess=False, attempts=10, outdir="datasets/")
