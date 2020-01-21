# predictive_model

This repository is an implementation of a regression predictive model. The model is sklearn compatible and it has passed sklearn estimator check.
It can be used together with all other functionalities in sklearn like GridSearch, get_params, set_params and others. 

 
\documentclass{article}
\usepackage[utf8]{inputenc}

\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\title{predictive model project}
\usepackage[margin=1in,footskip=0.25in]{geometry}

\begin{document}

\maketitle

\section{Introduction}



In this predictive model, gradient descent is used as optimization. The gradient descent minimizes the sum of residuals  $R(\alpha,\beta):=\sum_{i=1}^N L(f(x_i|\alpha, \beta),y_i)$.

\subsection{Derivative of Loss function with respect to parameters}
Assuming we have train samples $(x_1,y_1),…,(x_N,y_N)$ and loss function 
$L(y,y’):= |y-y’|^a$,  where $\hat{y} = f(x| \alpha, \beta)$.
And $f(x| \alpha, \beta) = \beta X + \alpha$,  Then the prediction of $\hat{y}$ can be written as $$ \hat{y} = \alpha+ \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n $$


letting $\alpha =  \beta_0 x_0$, then we can rewrite $\hat{y}$ as below 

\begin{align}\label{eq1}
 \hat{y} &= \beta_0 x_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n \\
	&   ~~~~x_0 =  1 \nonumber	
\end{align}.

equation (\ref{eq1}) can be written in matrix form as
\begin{align}\label{eq_2}
\hat{y} = \beta ^T X
\end{align}



\begin{align}\label{eq3_error}
L(y,\beta^T X):= |y-\beta ^T X|^a
\end{align}

We need to minimize the error in equation (\ref{eq3_error}) w.r.t $\beta$.

\begin{align}\label{eq_deri}
\frac{\partial L(y,\beta^T X) }{\partial \beta} &= a |y-\beta ^T X|^{a-1} ~\frac{\partial |y-\beta ^T X|}{\partial\beta}
\end{align}

we know that given $y = |x|$,~ $\frac{d y }{d x}=  \ \frac{x}{|x|}$ then equation (\ref{eq_deri}) becomes


\begin{align}\label{eq_derivative}
\frac{\partial L(y,\beta^T X) }{\partial \beta} &= a |y-\beta ^T X|^{a-1} ~\frac{\partial (|y-\beta ^T X|)}{\partial\beta}\\\nonumber
&= a |y-\beta ^T X|^{a-1} ~ . \frac{(y-\beta ^T X)}{|y-\beta ^T X|} (-X)\\\nonumber
&= -aX \frac{|y-\beta ^T X|^{a} }{(y-\beta ^T X)^2}(y-\beta ^T X) 
\text { ----- This simplifies to }\\\nonumber
& = -aX\frac{|y-\beta ^T X|^{a}}{(y-\beta ^T X)} 
\end{align}


when $a> 2$ and $y-\beta^T X $ is large, the derivative of the cost function goes to infinity, which means there is no learning. So, from my analysis, I highly recommend using a value of $a$ between [1,3)
\end{document}

