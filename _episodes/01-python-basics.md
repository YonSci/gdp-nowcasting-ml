---
title: Python Basics
teaching: 1
exercises: 1
questions:
- "What is Python, and why is it popular?"
- "How do you declare a variable in Python?"
- "What are the basic data types in Python?"
- "What is the difference between `if`, `elif`, and `else` statements?"
- "How do you define a function in Python?"
objectives:
- "Understand Python's role in programming and its applications."
- "Learn the basics of Python syntax and structure."
- "Explore Python data types, variables, and operators."
- "Write and execute Python scripts for simple tasks."
- "Practice debugging and handling errors."
keypoints:
- "Python is a versatile and beginner-friendly programming language."
- "Variables and data types are the building blocks of Python programming."
- "Conditional statements and loops are essential for controlling program flow."
- "Functions help organize reusable blocks of code."
- "Python has a rich standard library and supports external libraries for advanced functionalities."

---

# Python Basics

## Training Agenda

1. **Introduction to Python**
   
### What is Python?

- Python is a **beginner-friendly** (syntax is simple and intuitive, similar to plain English) **versatile**, & **High-Level Language** that is widely used across various fields, including web development, data analysis, artificial intelligence, and more.
  
- It was created in the late 1980s by **Guido van Rossum** and emphasizes **readability** and **simplicity**, making it one of the most popular languages today.

### Key Features of Python:

#### Versatility/Multi-Purpose 
- Python can be used for wide range of applications:
   - Building websites.
   - Automating tasks.
   - Analyzing data.
   - Training machine learning models.

#### In-Demand Skill
  - Python ranks among the most popular programming languages in the world.
  - Companies like Google, Netflix, and NASA use Python, making it a valuable skill in the job market.
    
#### Rich Ecosystem/Extensive Libraries
- Python has thousands of libraries and frameworks to help you:
   - Work with data: Pandas, NumPy.
   - Create visualizations: Matplotlib, Seaborn.
   - Build machine learning models: Scikit-learn, TensorFlow.

#### Collaboration and Community Support/Large Community
- With millions of developers worldwide, you’ll find tutorials, resources, and libraries for almost any task.
- Stuck on a problem? Resources like Stack Overflow and Python forums have answers.

#### Open source
 - Python is an open-source language, meaning it’s freely available for use and modification.

#### Platform-independent
- Python code can run on multiple operating systems, including Windows, Linux, and Mac operating systems.

#### "Hello, World!" in Different Languages

Assembly Language
```
section  data
    hello db 'Hello, World!',10
    len equ $ - hello

section  text
    global _start

_start:
    ; write our string to stdout
    mov eax, 4         ; sys_write
    mov ebx, 1         ; file descriptor 1 (stdout)
    mov ecx, hello     ; message to write
    mov edx, len       ; message length
    int 0x80           ; syscall
    ; exit
    mov eax, 1         ; sys_exit
    xor ebx, ebx       ; exit status 0
    int 0x80           ; syscall
```

Java Language

```java 
public class HelloWorld {
    public static void main(String[] args) {
        System out println("Hello, World!");
    }
}
```

C Language

```c
#include <stdio h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

R Language
```python 
print("Hello, World!")
```

Python Language
```python 
print("Hello, World!")
```

#### Most Popular Programming Languages from 1965- 2024
<iframe width="560" height="315" src="https://www.youtube.com/embed/xOW3Cehg_qg?si=dR6bCJjmc6yTFPR_" frameborder="0" allowfullscreen></iframe>

### Downsides of Python
  
  - It's `slow`:
    - Python code is interpreted during run time using CPython (line by line) rather than being compiled and executed by the compiler such as Java, C, C++, & Frotran.
    - Dynamically typed language

---




2. **Basic Syntax and Structure**
   - Writing your first Python program.
   - Python indentation rules.
   - Comments and documentation.

3. **Variables and Data Types**
   - Declaring variables.
   - Exploring data types: `int`, `float`, `str`, `bool`, and `list`.

4. **Operators**
   - Arithmetic, comparison, and logical operators.
   - Simple arithmetic operations.

5. **Control Structures**
   - `if`, `elif`, and `else` statements.
   - Loops: `for` and `while`.

6. **Functions**
   - Defining and calling functions.
   - Function arguments and return values.

7. **Error Handling**
   - Common Python errors.
   - Debugging basics.

8. **Conclusion and Wrap-Up**
   - Recap of key concepts.
   - Additional resources for further learning.

---

## Exercise
1. Write a script to calculate the area of a rectangle, given its length and width.
2. Create a loop that prints all the odd numbers between 1 and 20.
3. Define a function that takes two numbers and returns their sum.

---



