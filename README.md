# calculator
实现一个自动生成小学四则运算题目的命令行程序
<br />
## 生成题目和答案
输入`python arithmetic.py -20`，将会生成20道题目，题目文件存入Exercises.txt中
<br /><br />
题目范围限制：
输入`python arithmetic.py -r 10`，将会生成**10**以内的四则运算题目
<br /><br />
答案存入文件Answers.txt
## 批改答案
输入`python arithmetic.py -e Exercisea.txt -a Answers.txt`，将会对答案进行批改，对错结果存入文件Grade.txt中
