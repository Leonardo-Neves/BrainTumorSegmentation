import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('metrics.csv', sep=';')

# dataframe['Similarity'] = dataframe['Similarity'].str.replace(',', '.').astype(float)
# dataframe['Percentage Over Original Mask Total Area'] = dataframe['Percentage Over Original Mask Total Area'].str.replace(',', '.').astype(float)


# print('Number of images segmented: ', dataframe[dataframe['Segmented'] == True]['Image'].count())

# print('Number of images segmented - Coronal: ', dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Coronal')]['Image'].count())
# print('Number of images segmented - Axial: ', dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Axial')]['Image'].count())
# print('Number of images segmented - Sagittal: ', dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Sagittal')]['Image'].count())

# print('Number of images was not segmented: ', dataframe[dataframe['Segmented'] == False]['Image'].count())


# number_images_segmented_coronal = dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Coronal')]['Image'].count()
# number_images_segmented_axial = dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Axial')]['Image'].count()
# number_images_segmented_sagittal = dataframe[(dataframe['Segmented'] == True) & (dataframe['Perspective'] == 'Sagittal')]['Image'].count()

# plt.bar(['Coronal', 'Axial', 'Sagittal'], [number_images_segmented_coronal, number_images_segmented_axial, number_images_segmented_sagittal])
# plt.xlabel('Perspectiva')
# plt.ylabel('Número de imagens segmentadas')
# plt.title('Numero de imagens segmentadas por perspectiva')

# plt.show()

# print('Imagem não segmentada: ', dataframe[dataframe['Segmented'] == False]['Image'].count())
# print('Imagem segmentada: ', dataframe[dataframe['Segmented'] == True]['Image'].count())

# plt.bar(['Imagem não segmentada', 'Imagem segmentada'], [dataframe[dataframe['Segmented'] == False]['Image'].count(), dataframe[dataframe['Segmented'] == True]['Image'].count()])
# plt.ylabel('Número de imagens utilizadas')
# plt.title('Relação entre imagens segmentadas e não segmentadas')

# plt.show()

print('Jaccard Index Axial: ', dataframe[dataframe['Jaccard Index Axial'] != 0]['Jaccard Index Axial'].count())
print('Jaccard Index Coronal: ', dataframe[dataframe['Jaccard Index Coronal'] != 0]['Jaccard Index Coronal'].count())
print('Jaccard Index Sagittal: ', dataframe[dataframe['Jaccard Index Sagittal'] != 0]['Jaccard Index Sagittal'].count())

print('Panoptic Quality Axial: ', dataframe[dataframe['Panoptic Quality Axial'] != 0]['Panoptic Quality Axial'].count())
print('Panoptic Quality Coronal: ', dataframe[dataframe['Panoptic Quality Coronal'] != 0]['Panoptic Quality Coronal'].count())
print('Panoptic Quality Sagittal: ', dataframe[dataframe['Panoptic Quality Sagittal'] != 0]['Panoptic Quality Sagittal'].count())

print('Dice Coefficient Axial: ', dataframe[dataframe['Dice Coefficient Axial'] != 0]['Dice Coefficient Axial'].count())
print('Dice Coefficient Coronal: ', dataframe[dataframe['Dice Coefficient Coronal'] != 0]['Dice Coefficient Coronal'].count())
print('Dice Coefficient Sagittal: ', dataframe[dataframe['Dice Coefficient Sagittal'] != 0]['Dice Coefficient Sagittal'].count())

print('Jaccard Index Axial: ', dataframe[dataframe['Jaccard Index Axial'] != 0]['Jaccard Index Axial'].mean())
print('Jaccard Index Coronal: ', dataframe[dataframe['Jaccard Index Coronal'] != 0]['Jaccard Index Coronal'].mean())
print('Jaccard Index Sagittal: ', dataframe[dataframe['Jaccard Index Sagittal'] != 0]['Jaccard Index Sagittal'].mean())

print('Panoptic Quality Axial: ', dataframe[dataframe['Panoptic Quality Axial'] != 0]['Panoptic Quality Axial'].mean())
print('Panoptic Quality Coronal: ', dataframe[dataframe['Panoptic Quality Coronal'] != 0]['Panoptic Quality Coronal'].mean())
print('Panoptic Quality Sagittal: ', dataframe[dataframe['Panoptic Quality Sagittal'] != 0]['Panoptic Quality Sagittal'].mean())

print('Dice Coefficient Axial: ', dataframe[dataframe['Dice Coefficient Axial'] != 0]['Dice Coefficient Axial'].mean())
print('Dice Coefficient Coronal: ', dataframe[dataframe['Dice Coefficient Coronal'] != 0]['Dice Coefficient Coronal'].mean())
print('Dice Coefficient Sagittal: ', dataframe[dataframe['Dice Coefficient Sagittal'] != 0]['Dice Coefficient Sagittal'].mean())

# print(dataframe.head(10))