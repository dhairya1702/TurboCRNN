# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:15:46 2022

@author: 91805
"""


import os, random, shutil

#Prompting user to enter number of files to select randomly along with directory
source=input("Enter the Source Directory : ")
#dest=input("Enter the Destination Directory : ")
#no_of_files=int(input("Enter The Number of Files To Select : "))

dest = "D:\Work\VIT\Capstone Project\Dataset\False Samples"
no_of_files = 61

print("%"*25+"{ Details Of Transfer }"+"%"*25)
print("\n\nList of Files Moved to %s :-"%(dest))

#Using for loop to randomly choose multiple files
for i in range(no_of_files):
    #Variable random_file stores the name of the random file chosen
    random_file=random.choice(os.listdir(source))
    print("%d} %s"%(i+1,random_file))
    source_file="%s/%s"%(source,random_file)
    dest_file=dest
    #"shutil.move" function moves file from one directory to another
    shutil.move(source_file,dest_file)

print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)