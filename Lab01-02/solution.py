import copy as cp
##################################################################
#                       Domain Element                           #
##################################################################
class DomainElement():

    #Class constructor
    def __init__(self, list):

        self.values = list

    # Function that returns number of components in a element - element might look like (1) or (1,2)...
    def getNumberOfComponents(self):

        return len(self.values)

    # Function that returns component value at given index
    def getComponentValue(self, index):

        return self.values[index]

    #Definition of element comparator
    def __eq__(self, other):

        #First check sizes
        if self.getNumberOfComponents() != other.getNumberOfComponents():
            return False

        #Then compare each component
        for x,y in zip(self.values, other.values):
            if x != y:
                return False
        
        #Return true if elements are same
        return True

    #To string function
    def __str__(self):
        
        string = "("

        for element in self.values:
            string += str(element)+","
        
        string = string[:-1]
        string += ")"

        return string

    #static method for creating Domain elements
    @staticmethod
    def of(list):
        return DomainElement(list)


##################################################################
#                       Abstract Domain                          #
##################################################################
class Domain():

    #Creates a SimpleDomain - SimpleDomain consists of single elements (1),(2),(3)...
    @staticmethod
    def intRange(lower, upper):
    
        return SimpleDomain(lower,upper)

    #Creates a CompositeDomain - CompositDomain consist of complex elements (1,1),(1,2)(2,1)...
    @staticmethod
    def combine(*domains):

        return CompositeDomain(domains)

    #Returns index of given element -REQUIRES DomainElement to be given!!!!
    def indexOfElement(self,element):

        index = 0
        for elem in self:
            if element == elem:
                return index
            else:
                index += 1

    #Returns DomainElement for given index
    def elementForIndex(self,index):
        
        count = 0

        for elem in self:
            if count == index:
                return elem
            else:
                count += 1

    #Fake iterator for Domain class - instances of Domain class are never going to be created, this is just a fix
    def __iter__(self):

        return iter([])

     #Return cardinal number of domain
    def getCardinality(self):
        pass

    def __str__(self):
        for elem in self:
            print(elem)
        
        print("Kardinalite domene je {}".format(self.getCardinality()))
        return ""


##################################################################
#                        Simple Domain                           #
##################################################################
class SimpleDomain(Domain):


    def __init__(self,first,last):
        self.first = first
        self.last = last
    
    #Return cardinal number of domain
    def getCardinality(self):

        return self.last-self.first

    #Returns self as a component - SimpleDomain has only one(self) component
    def getComponent(self, integer):
        return self

    #Returns 1 - SimpleDomain has only one(self) component
    def getNumberOfComponents(self):
        return 1

    #Making a class iterable
    def __iter__(self):

        domain = []
        for element in range(self.first,self.last):
            domain.append(DomainElement([element]))
        return iter(domain)


##################################################################
#                       Composite Domain                         #
##################################################################
class CompositeDomain(Domain):

    def __init__(self, *args):

        self.domains = []
        for domain in args[0]:
            self.domains.append(domain)

    #Calculates and returns cardinality of a domain by multiplying cardinalities of components
    def getCardinality(self):

        cardinality = 1

        for domain in self.domains:
            cardinality *= domain.getCardinality()

        return cardinality

    #Returns SimpleDomain component at given index
    def getComponent(self, index):
        return self.domains[index]

    #Returns number of components of a CompositeDomain
    def getNumberOfComponents(self):
        return len(self.domains)

    #Making a class iterable
    def __iter__(self):

        elements = []
        tempElements = []

        for domain in self.domains:

            if len(elements) == 0:

                for elem in range(domain.first,domain.last):
                    elements.append([elem])
            else:
                for elem in elements:
                    for elem2 in domain:
                        tempElements.append(elem+elem2.values)

                elements = tempElements.copy()
                tempElements.clear()
        domain = []

        for elem in elements:
            domain.append(DomainElement(elem))

        return iter(domain)

        

#Primjer sa labosa 1.DIO
d1 = Domain.intRange(0,5)
print("Elementi domene d1:")
print(d1)

d2 = Domain.intRange(0,3)
print("Elementi domene d2:")
print(d2)

d3= Domain.combine(d1,d2)
print("Elementi domene d3:")
print(d3)

print(d3.elementForIndex(0))
print(d3.elementForIndex(5))
print(d3.elementForIndex(14))
print(d3.indexOfElement(DomainElement.of([4,1])))



##################################################################
#                      Mutable Fuzzy Set                         #
##################################################################

class MutableFuzzySet():

    #Konstruktor
    def __init__(self,domain):

        self.domain = domain
        self.membership = [0.0]*domain.getCardinality()

    #Provjeri vrijednost vrijednosne funkcije za određeni element - DomainElement required to bi given
    def getValueAt(self,element):
        
        index = self.domain.indexOfElement(element)

        return self.membership[index]

    #Postavi vrijednost vrijednosne funckije za određeni element
    def set(self, element, value):

        index = self.domain.indexOfElement(element)

        self.membership[index] = value

    #Funkcija za printanje
    def print(self):

        for elem in self.domain:
            print("d{}={}".format(elem,self.getValueAt(elem)))

##################################################################
#                    Calculated Fuzzy Set                        #
##################################################################

class CalculatedFuzzySet():

    #Konstruktor
    def __init__(self,domain, function):

        self.domain = domain
        self.membership = [0.0]*domain.getCardinality()
        self.function = function

    #Izracunaj vrijednost vrijednosne funkcije
    def getValueAt(self, element):
        
        return self.function.valueAt(element.values)

    #Funkcija za printanje
    def print(self):

        for elem in self.domain:
            print("d{}={}".format(elem,self.getValueAt(elem)))


##################################################################
#                       LFunction class                          #
##################################################################

class LFunction():

    def __init__(self,alpha,betta):

        self.alpha = alpha
        self.betta = betta

    def valueAt(self, x):

        if x[0] <= self.alpha:
            return 1.0
        elif x[0] >= self.betta:
            return 0.0
        else:
            return (self.betta-x[0])/(self.betta-self.alpha)

##################################################################
#                     GammaFunction class                        #
##################################################################

class GammaFunction():

    def __init__(self,alpha,betta):

        self.alpha = alpha
        self.betta = betta

    def valueAt(self, x):

        if x[0] <= self.alpha:
            return 0.0
        elif x[0] >= self.betta:
            return 1.0
        else:
            return (x[0]-self.alpha)/(self.betta-self.alpha)

##################################################################
#                     LambdaFunction class                       #
##################################################################

class LambdaFunction():

    def __init__(self,alpha,betta,gamma):

        self.alpha = alpha
        self.betta = betta
        self.gamma = gamma

    def valueAt(self, x):

        if x[0] <= self.alpha:
            return 0.0
        elif x[0] >= self.gamma:
            return 0.0
        elif self. betta > x[0] > self.alpha:
            return (x[0]-self.alpha)/(self.betta-self.alpha)
        else:
            return (self.gamma-x[0])/(self.gamma-self.betta)

print("")
#Primjer sa labosa 2.DIO
d = Domain.intRange(0,11)

set1 = MutableFuzzySet(d)
set1.set(DomainElement.of([0]), 1.0)
set1.set(DomainElement.of([1]), 0.8)
set1.set(DomainElement.of([2]), 0.6)
set1.set(DomainElement.of([3]), 0.4)
set1.set(DomainElement.of([4]), 0.2)

print("Set1:")
set1.print()
print("")

d2 = Domain.intRange(-5,6)
set2 = CalculatedFuzzySet(d2,LambdaFunction(-4,0,4))
print("Set2:")
set2.print()
print("")

##################################################################
#                          Operations                            #
##################################################################
class Operations():

    @staticmethod
    def zadehNot(fuzzySet):

        copySet = cp.deepcopy(fuzzySet)
        copySet.getValueAt = lambda x: 1-fuzzySet.getValueAt(x)

        return copySet

    @staticmethod
    def zadehAnd(set1,set2):

        copySet = cp.deepcopy(set1)
        copySet.getValueAt = lambda x: min(set1.getValueAt(x),set2.getValueAt(x))

        return copySet

    @staticmethod
    def zadehOr(set1,set2):

        copySet = cp.deepcopy(set1)
        copySet.getValueAt = lambda x: max(set1.getValueAt(x),set2.getValueAt(x))

        return copySet

    @staticmethod
    def hamacherTNorm(set1,set2,param):

        copySet = cp.deepcopy(set1)
        copySet.getValueAt = lambda x: (set1.getValueAt(x)*set2.getValueAt(x))/(param+(1-param)*(set1.getValueAt(x)+set2.getValueAt(x)-set1.getValueAt(x)*set2.getValueAt(x)))

        return copySet

    @staticmethod
    def hamacherSNorm(set1,set2,param):

        copySet = cp.deepcopy(set1)
        copySet.getValueAt = lambda x: (set1.getValueAt(x)+set2.getValueAt(x)-(2-param)*(set1.getValueAt(x)*set2.getValueAt(x)))/(1-(1-param)*(set1.getValueAt(x)*set2.getValueAt(x)))

        return copySet


    

#Primjer sa labosa 3.DIO
d = Domain.intRange(0,11)

set1 = MutableFuzzySet(d)
set1.set(DomainElement.of([0]), 1.0)
set1.set(DomainElement.of([1]), 0.8)
set1.set(DomainElement.of([2]), 0.6)
set1.set(DomainElement.of([3]), 0.4)
set1.set(DomainElement.of([4]), 0.2)

print("Set1:")
set1.print()
print("")

notSet1 = Operations.zadehNot(set1)
print("notSet1:")
notSet1.print()
print("")

union = Operations.zadehOr(set1,notSet1)
print("Set1 union notSet1:")
union.print()
print("")

hinters = Operations.hamacherTNorm(set1,notSet1,1.0)
print("Set1 intersection with notSet1 using parameterised Hamacher T norm with parameter 1.0:")
hinters.print()
print("")


##################################################################
#                          Relations                             #
##################################################################


class Relations():

    @staticmethod
    def isUtimesURelation(relation):
        if relation.domain.getNumberOfComponents() == 2:
            if relation.domain.domains[0].first == relation.domain.domains[1].first and relation.domain.domains[0].last == relation.domain.domains[1].last:
                return True
        else:
            return False

    @staticmethod
    def isSymmetric(relation):
        for x in range(relation.domain.domains[0].first,relation.domain.domains[0].last):
            for y in range(relation.domain.domains[1].first,relation.domain.domains[1].last):
                if relation.getValueAt(DomainElement.of([x,y])) != relation.getValueAt(DomainElement.of([y,x])):
                    return False
        return True

    @staticmethod
    def isReflexsive(relation):
        for x in range(relation.domain.domains[0].first,relation.domain.domains[0].last):
            if relation.getValueAt(DomainElement.of([x,x])) != 1:
                return False
        return True

    @staticmethod
    def isMaxMinTransitive(relation):
        for x in range(relation.domain.domains[0].first,relation.domain.domains[0].last):
            for z in range(relation.domain.domains[1].first,relation.domain.domains[1].last):
                maks = []
                for y in range(relation.domain.domains[0].first,relation.domain.domains[0].last):
                    maks.append(min(relation.getValueAt(DomainElement.of([x,y])),relation.getValueAt(DomainElement.of([y,z]))))
                maks = max(maks)

                if maks > relation.getValueAt(DomainElement.of([x,z])):
                    return False
        return True

    @staticmethod
    def compositionOfBinaryRelations(relation1,relation2):
        u1 = relation1.domain.domains[0]
        u2 = relation2.domain.domains[1]
        u3 = Domain.combine(u1,u2)
        relation = MutableFuzzySet(u3)

        for x in range(relation.domain.domains[0].first,relation.domain.domains[0].last):
            for z in range(relation.domain.domains[1].first,relation.domain.domains[1].last):
                maks = []
                for y in range(relation1.domain.domains[1].first,relation1.domain.domains[1].last):
                    maks.append(min(relation1.getValueAt(DomainElement.of([x,y])),relation2.getValueAt(DomainElement.of([y,z]))))

                maks = max(maks)

                relation.set(DomainElement.of([x,z]),maks)

        return relation

    @staticmethod
    def isFuzzyEquivalence(relation):
        if Relations.isReflexsive(relation) and Relations.isSymmetric(relation) and Relations.isMaxMinTransitive(relation):
            return True
        return False



#Primjer sa labosa 1.DIO

u = Domain.intRange(1,6)
u2 = Domain.combine(u,u)

r1 = MutableFuzzySet(u2)
r1.set(DomainElement.of([1,1]),1)
r1.set(DomainElement.of([2,2]),1)
r1.set(DomainElement.of([3,3]),1)
r1.set(DomainElement.of([4,4]),1)
r1.set(DomainElement.of([5,5]),1)
r1.set(DomainElement.of([3,1]),0.5)
r1.set(DomainElement.of([1,3]),0.5)

r2 = MutableFuzzySet(u2)
r2.set(DomainElement.of([1,1]),1)
r2.set(DomainElement.of([2,2]),1)
r2.set(DomainElement.of([3,3]),1)
r2.set(DomainElement.of([4,4]),1)
r2.set(DomainElement.of([5,5]),1)
r2.set(DomainElement.of([3,1]),0.5)
r2.set(DomainElement.of([1,3]),0.1)

r3 = MutableFuzzySet(u2)
r3.set(DomainElement.of([1,1]),1)
r3.set(DomainElement.of([2,2]),1)
r3.set(DomainElement.of([3,3]),0.3)
r3.set(DomainElement.of([4,4]),1)
r3.set(DomainElement.of([5,5]),1)
r3.set(DomainElement.of([1,2]),0.6)
r3.set(DomainElement.of([2,1]),0.6)
r3.set(DomainElement.of([2,3]),0.7)
r3.set(DomainElement.of([3,2]),0.7)
r3.set(DomainElement.of([3,1]),0.5)
r3.set(DomainElement.of([1,3]),0.5)

r4 = MutableFuzzySet(u2)
r4.set(DomainElement.of([1,1]),1)
r4.set(DomainElement.of([2,2]),1)
r4.set(DomainElement.of([3,3]),1)
r4.set(DomainElement.of([4,4]),1)
r4.set(DomainElement.of([5,5]),1)
r4.set(DomainElement.of([1,2]),0.4)
r4.set(DomainElement.of([2,1]),0.4)
r4.set(DomainElement.of([2,3]),0.5)
r4.set(DomainElement.of([3,2]),0.5)
r4.set(DomainElement.of([3,1]),0.4)
r4.set(DomainElement.of([1,3]),0.4)

print("r1 je definiran nad UxU?",Relations.isUtimesURelation(r1))
print("r1 je simetrična",Relations.isSymmetric(r1))
print("r2 je simetrična",Relations.isSymmetric(r2))
print("r1 je refleksivna?",Relations.isReflexsive(r1))
print("r3 je refleksivna?",Relations.isReflexsive(r3))
print("r3 je max-min tranzitivna?",Relations.isMaxMinTransitive(r3))
print("r4 je max-min tranzitivna?",Relations.isMaxMinTransitive(r4))

#Primjer sa labosa 2.DIO

u1 = Domain.intRange(1,5)
u2 = Domain.intRange(1,4)
u3 = Domain.intRange(1,5)

r1 = MutableFuzzySet(Domain.combine(u1,u2))
r1.set(DomainElement.of([1,1]),0.3)
r1.set(DomainElement.of([1,2]),1)
r1.set(DomainElement.of([3,3]),0.5)
r1.set(DomainElement.of([4,3]),0.5)

r2 = MutableFuzzySet(Domain.combine(u2,u3))
r2.set(DomainElement.of([1,1]), 1)
r2.set(DomainElement.of([2,1]), 0.5)
r2.set(DomainElement.of([2,2]), 0.7)
r2.set(DomainElement.of([3,3]), 1)
r2.set(DomainElement.of([3,4]), 0.4)

r1r2 = Relations.compositionOfBinaryRelations(r1,r2)
r1r2.print()

#Primjer sa labosa 3.DIO

u = Domain.intRange(1,5)

r = MutableFuzzySet(Domain.combine(u,u))
r.set(DomainElement.of([1,1]), 1)
r.set(DomainElement.of([2,2]), 1)
r.set(DomainElement.of([4,4]), 1)
r.set(DomainElement.of([3,3]), 1)
r.set(DomainElement.of([1,2]), 0.3)
r.set(DomainElement.of([2,1]), 0.3)
r.set(DomainElement.of([2,3]), 0.5)
r.set(DomainElement.of([3,2]), 0.5)
r.set(DomainElement.of([3,4]), 0.2)
r.set(DomainElement.of([4,3]), 0.2)

r2 = r

print("Početna relacija je neizrazita relacija ekvivalencije?",Relations.isFuzzyEquivalence(r2))

for i in range(1,4):
    print("")
    r2 = Relations.compositionOfBinaryRelations(r2,r)

    print("Broj odrađenih kompozicija je {}, Relacija je:".format(i))
    r2.print()
    print("Ova relacija je neizrazita relacija ekvivalencije?",Relations.isFuzzyEquivalence(r2))