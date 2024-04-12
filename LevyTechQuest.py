import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm

def partA(data):
    mean = sum(a for a in data)/len(data)                    
    variance = sum([(a - mean)**2 for a in data]) / len(data)  
    stdDev = variance ** (1/2)
    numEle = len(data)
    
    return mean, stdDev, numEle

def partB(data, threshold):
    mean, stdDev, n = partA(data)
    zScore = (mean - threshold) / (stdDev/n ** (1/2))
    pVal = 1 - norm.cdf(zScore)
    return zScore, pVal

class GeneExpression:
    def __init__(self, csvFile, threshold):
        self.threshold = threshold
        self.data = self.loadData(csvFile)
    
    def loadData(self, csvFile):
        data = {}
        with open(csvFile, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                gene = row[0]
                expVal = [float(v) for v in row[1:len(row)-3]]
                print(expVal)
                data[gene] = expVal
        return data

    def partC(self):
        results = {}
        for gene, expVal in self.data.items():
            zScore, pVal = partB(expVal, self.threshold) 
            mean = sum(expVal) / len(expVal)
            test = 'bad' if mean > self.threshold else 'good' 
            results[gene] = {'pVal': pVal, 'zScore' : zScore, 'geneType': test}
        return results
    
    def partD(self, gene1, gene2):
        plt.scatter(self.data[gene1], self.data[gene2])
        plt.title("Scatter plot of " + gene1 + "and " + gene2)
        plt.xlabel(gene1 + " expressions")
        plt.ylabel(gene2 + " expressions")
        plt.show()


# part A and part B
data = [5.99342831, 4.7234714, 6.29537708, 8.04605971, 4.53169325,
        4.53172609, 8.15842563, 6.53486946, 4.06105123, 6.08512009]
threshold = 4
mean, stdDev, numEle = partA(data)
zScore, pVal = partB(data, threshold)
print("part A and part B")
print("Mean:", mean)
print("Standard Deviation:", stdDev)
print("Number of Elements:", numEle)
print("Z score:", zScore)
print("P value:", pVal)
print()

# part C
#Change csvfilepath based on the path your data is stored
csvfilepath = r"sampleData.csv"
analyzer = GeneExpression(csvfilepath, threshold)
results = analyzer.partC()
print("part C")
for gene, result in results.items():
    print(gene)
    print("p-value: ", result['pVal'])
    print("z-score: ", result['zScore'])
    print("Gene type: ", result['geneType'])

# part D
gene1 = 'G1'
gene2 = 'G2'
analyzer.partD(gene1, gene2)