import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('Salary_Data.csv');
x = data.iloc[: ,0:1].values
y = data.iloc[:, 1].values


#no imputing recquired
#no encoding as there are no categorical data

#SPLITTING INTO TEST AND TTAINING
xtraining, xtest, ytraining, ytest = train_test_split(x, y, test_size=0.25, random_state=1)

class Mycode:
    alpha = 0.01
    theta1 = 1
    theta0 = 5
    errorchange = 1
    previouscost = -10

    currentcost = 0
    gradient0, gradient1 = 0, 0
    costs = []



    def computecost(this):
        i = 0
        errortotal = 0
        temp = this.currentcost

        while i<len(xtraining):
            errortotal += (this.theta0 + this.theta1*xtraining[i]-ytraining[i])**2
            i+=1


        this.currentcost = (errortotal)/(2*len(xtraining))
        this.costs.append(this.currentcost)
        this.previouscost = temp
        return this.currentcost, this.previouscost

    def findgradient(self):
        i = 0
        tempgradient0, tempgradient1 = 0, 0
        while i<len(xtraining):
            tempgradient0+= self.theta0 + self.theta1*xtraining[i]-ytraining[i]
            tempgradient1+=(self.theta0+self.theta1*xtraining[i]-ytraining[i])*xtraining[i]
            i+=1

        tempgradient0 = (1/len(xtraining))*tempgradient0
        tempgradient1 /= len(xtraining)
        self.gradient0 = tempgradient0
        self.gradient1 = tempgradient1

    def updatetheta (self):
        self.theta0 = self.theta0 - self.alpha*self.gradient0
        self.theta1-= self.alpha*self.gradient1

    def converge (self):
        return abs((self.currentcost-self.previouscost))<self.errorchange


    def run (self):
        i = 0
        while i<100000 and not self.converge():
            i = i+1
            self.computecost()
            self.findgradient()
            self.updatetheta()
            # print(self.currentcost-self.previouscost)
        if i==100000:
            print("ran out of attempts")
        else:
            print("converged in less than {} iterations".format(i+1))

    def plot (self):
        mpl.scatter(xtraining, ytraining, color = '#0004ff')
        mpl.plot(xtraining, self.theta0+self.theta1*xtraining, color = '#ff0000')
        mpl.show()
    def costplotter(self):
        mpl.plot(self.costs[0:5:1])
        mpl.show()

    def predictor(self, xtest):
        return self.theta0+self.theta1*xtest






mycode = Mycode()
mycode.run()
ypredicted = mycode.predictor(xtest)
print(r2_score(ytest, ypredicted))
for i in range(0, len(ypredicted),1):
    print('{}    {}'.format(ypredicted[i], ytest[i]))

mycode.costplotter()





