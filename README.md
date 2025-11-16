# **Hi-Lo-Tip-Predictor**

A program that uses a naive bayes classifier (also contains some continuous features) to predict, based on which day of the week it is, how many people are at dinner, if it is a male/female tipper if they are going to tip high (i.e. >15%) or low.

## **How It Works**

The program uses basic python features to store the data (dictionaries and lists) and numpy for fast math calculations on arrays.

It takes in the csv, parses it and splits the data (separated by a comma) into an array of parameters.

## **My learnings and contributions**

Even though the app has only ~150 lines of code as of writing (will probably remain like that 😒) it is dense with information. I learned how laplace smoothing works (basically just use logarihms because sums are better than products, and computers are not good with numbers extremely small).

The program is weird beccause if we change a tiny number, the accuracy can skyrocket, but the data would be flawed I assume. What I mean by this is that if we change what we consider as a high tip (let's say a high tip is 25% or more extra) we would get an above 90% accuracy easily, because almost no data would be high tipping anymore. I chose a middle point, 15%, that gives us a realistic accuracy of like 63-64%.

I did not test what accuracies I would get without Laplace smoothing, sorry.

One of the **biggest** thing I found out is that we can make more use of the data we already know by making more categories out of it. There is not "_is_weekend_" data in the csv file because it would obviously be irellevant. This addition granted us a whole +3% accuracy, which, for a naive bayes approach, is absoloutely fenomenal. I hope in the future I can make this entire project continuous with minimal effort, since I think the code is MUCH MUCH cleaner than the categorising thing, because it's just a formula. Computers like math, and they're freaking good at it.

Fun fact! Mathemathics wikipedia articles are very badly written. I am _TERRIBLE_ ad maths and don't expect anything from myself most of the time, but I do appreciate a 3blue1brown video every now and then. Some articles have the definitions use the word they are defining inside them... Yeah, chatgpt is a much better tool for learning if you know to ask him targeted questions and test his code/visit the links he "sourced the information from".

TL;DR:

- +3% gain from adding an "is_weekend" parameter;
- I used laplace smoothing but didnt record changes w/o it;
- Wikipedia bad, chatgpt good if you know how to use it properly;

## **Running the Program**

Requirements:

- Python 3
- NumPy
- Matplotlib (even though it's not used in the final version)

To run:

```bash
python3 main.py
```

It is a simple cli interface, and if you want to change some parameters you need to scavange my code, sorry.

## **Conclusion**

I think this project was cool, even though small, it made me appreciate the ammount of information that you can extract from something so simple.
