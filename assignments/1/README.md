# Assignment 1 Report

Question 2->
the perform eda task was done by LLM using the following prompt->
pasting task 2.1.1 and 2.1 in the prompt window

after numerous debugging attempts by refeeding error code i got it to work

after noticing the plots I figured out the distortion was due to unbalanced classes like population and duration_ms which made it 
hard to see the other data points , the first thing i did was remove the non numerical classes and then normalising the other classes
(the target_genre was encoded with the first unique type coding though this would cause issues with the KNN model cause the coding might differ from train set to test set depending which label came up as a unique first in those so i switched to one hot coding for the subsequent tasks)
the heirarchy of class importance was found by the pair wise correlation plot and the output was uploaded as a figure (1211 corr)

the KNN implementation was again done with the following prompt->
(2.3 KNN question pasted) this time along with the answer received I switched to one hot encoding for the track_genre so they are consistent in both the train and the val / test set 
2.4 was done with the prompt of the question being fed in the window, along with doing the splitting without use of sklearn, 
various errors popped up regarding the non numeric data being fed into the knn model , so to fix it i dropped the non numeric columns and forced the the true/false explicit into an integer 0 and 1 . then converted everything to float again , 

for the k best values i chose odd values at random from 1 - 120

the final result out of the values chosen was included in the figures folder {k = 71 ,eucledian}performed best



#3.1.1
done under a1_3.py
done by feeding the question into the llm prompt and then changing the file pathing ,
The first model given used the closed form solution which was disallowed later so the next model was done using gradient descent
the plots and MSE evaluation for simple regression case have been included in the figures folder
#3.1.2
The linear regression model remained the same , only the function polynomial_regression() was created with llm using the question in the prompt window
the final plots are in the figure

the model revealed the best degree was 3 learning rate [0.1] with MSE 0.0590 the parameter is stored with the figure read.me

next with #3.1.3 I tried using imageio and pillow to create a create gif function from llm but it was hard to integrate with the current linearregression class without breaking the first 2 tasks so I had to skip this one

#3.2.1
the model was the same regression model attained by using llm with the question as the prompt , refed the previous codes of the 2 tasks back to the llm so that they were working with this model.
The resulting mse,variance,etc measures are given in the figures , the plots are also there,
Observation - I didnt notice much overfitting with the model I used, as is evident that mse for k=20 no regularisation yielded me 0.01313 MSE
while the k = 20 regularisaion (L2) gave 0.01318 MSE.
The plots are separate for noreg L1 and L2 fits.

Thank you!.

For running the codes.
for task 2.X I have made all the tasks in the same file please just comment out the other tasks from the main driver accordingly
for tasks 3.X they are in a1_3 file , comment out the other tasks according to what is being tested in the main driver.
