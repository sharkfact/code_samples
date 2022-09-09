#Config file: toy Shiny dashboard with no senstive data
#Passwords are for demonstration purposes only

#Load packages
library(shiny)
library(shinydashboard)
library(dplyr)
library(DT)

#Passwords
users <- c("test", "dev")
passwords <- c("test", "test")