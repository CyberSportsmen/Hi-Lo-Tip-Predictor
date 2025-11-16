# **Hi-Lo-Tip-Predictor**

A program that uses a naive bayes classifier (also contains some continuous features) to predict, based on which day of the week it is, how many people are at dinner, if it is a male/female tipper if they are going to tip high (i.e. >15%) or low.

## **How It Works**

The program uses basic python features to store the data (dictionaries and lists) and numpy for fast math calculations on arrays.

It takes in the csv, parses it and splits the data (separated by a comma) into an array of parameters.

### A bit about each function (the code is too short not to write this)

- ReadData() → Opens the .csv file and parses it.
- SplitData() → splits the data into training data and testing data, with an implicit ratio of 80% training data.
- TrainParams() → Trains the "AI" parameters and returns them as dictionaries.
- Predict() → Given a data line, predicts if it is more likely to be high tipping or low tipping.
- Evaluate() → Just Takes the trained parameters and testing data and checks how many does the AI predict wrong/right
- main.py → just calls the functions above and takes the mean of 1000 evaluations to measure accuracy, then we print it out.

### **The Naive Bayes Model (very short explanation)**

For a given class $Y \in \{0,1\}$ and features $X_1, X_2, \dots, X_n$:

$$
P(Y \mid X_1, \dots, X_n) \propto P(Y) \prod_{i} P(X_i \mid Y)
$$

- $P(Y)$ = prior probability (how often high/low tips appear in the data)
- $P(X_i \mid Y)$ = likelihood of each feature given the class

For continuous features, we used the Gaussian probability density function:

$$
p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

This lets us model numeric things like total bill or group size smoothly.

### **Dataset Used**

The program uses the Seaborn Tips Dataset, available here:

👉 https://www.kaggle.com/datasets/ranjeetjain3/seaborn-tips-dataset

(You can download it as CSV and place it next to the script.)

---

## **My learnings and contributions**

Even though the app has only ~150 lines of code as of writing (will probably remain like that 😒) it is dense with information. I learned how laplace smoothing works (basically just use logarihms because sums are better than products, and computers are not good with numbers that are extremely small).

The program is weird beccause if we change a tiny number, the accuracy can skyrocket, but the data would be flawed I assume. What I mean by this is that if we change what we consider as a high tip (let's say a high tip is 25% or more extra) we would get an above 90% accuracy easily, because almost no data would be high tipping anymore. I chose a middle point, 15%, that gives us a realistic accuracy of like 63-64%.

I did not test what accuracies I would get without Laplace smoothing, sorry.

One of the **biggest** thing I found out is that we can make more use of the data we already know by making more categories out of it. There is not "_is_weekend_" data in the csv file because it would obviously be irellevant. This addition granted us a whole +3% accuracy, which, for a naive bayes approach, is absoloutely fenomenal. I hope in the future I can make this entire project continuous with minimal effort, since I think the code is MUCH MUCH cleaner than the categorising thing, because it's just a formula. Computers like math, and they're freaking good at it.

Fun fact! Mathemathics wikipedia articles are very badly written. I am _TERRIBLE_ ad maths and don't expect anything from myself most of the time, but I do appreciate a 3blue1brown video every now and then. Some articles have the definitions use the word they are defining inside them... Yeah, chatgpt is a much better tool for learning if you know to ask him targeted questions and test his code/visit the links he "sourced the information from".

TL;DR:

- Computers love numbers :D
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

## How to change stuff and tinker with the program

It is a simple cli interface, and if you want to change some parameters you need to scavange my code, sorry.

If you want to change what % is ok for a tip, ctrl+f on "0.15" and change that to whatever you want in both places. (TODO: add this as an argument maybe)

If you want to have a 50/50 split of tests and training, you can do that, just call the function SplitData with the paramenter ratio altered.

You can clearly see that the first line of the .csv file contains the formatting of data, respect it. Otherwise, you can add whatever you want there.

## **Refferences**

- Wikipedia page on Naive Bayes
- The Seaborn Tips dataset
- 3Blue1Brown (https://www.youtube.com/watch?v=HZGCoVF3YvM)
- The course slides

## **Conclusion**

I think this project was cool, even though small, it made me appreciate the ammount of information that you can extract from something so simple.

In retrospect, I should have chosen a bigger dataset, but I found out that extracting most out of a tiny dataset is a _much_ bigger challenge.

Thanks for reading! <3
