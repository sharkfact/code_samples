library(shiny)
library(rsconnect)
library(ggplot2)
library(dplyr)
library(reshape2)
library(plyr)
library(readr)
library(ggthemes)

# DATA CLEANING ----------------------------------------------------------------
#Import juvenile arrest rate data
jar <- read_csv("JAR.csv")

#Replace all n/a values with NA for R
jar[jar=="n/a"] <- NA

#Convert yearly counts columns to numeric
jar[, 3:38] <- sapply (jar[, 3:38], as.numeric)

#Melt the data
jar <- reshape2::melt(jar)

#Make column names interpretable
jar <- plyr::rename(jar, c("variable" = "Year", "value" = "PropPer100k"))

#Recode Offense column to have fewer, more logical levels
jar$Offense <- as.factor(jar$Offense)
levels(jar$Offense) <- list("Violent crimes" = c("Violent Crime",
                                                 "Murder and nonnegligent manslaughter",
                                                 "Forcible rape",
                                                 "Aggravated assault",
                                                 "Other assaults"),
                            "Robbery and theft" = c("Burglary",
                                                    "Larceny-theft",
                                                    "Motor vehicle theft",
                                                    "Property Crime Index",
                                                    "Robbery"),
                            "Arson" = "Arson",
                            "Runaways" = "Runaways",
                            "Weapons possession" = "Weapons carrying, possessing",
                            "Drug abuse violations" = "Drug abuse violations",
                            "Vandalism and loitering" = c("Vandalism",
                                                          "Curfew and loitering law violations"),
                            "Liquor violations and disorderly conduct" = c("Liquor laws",
                                                                           "Drunkenness",
                                                                           "Disorderly conduct",
                                                                           "Driving under the influence"),
                            "All crimes" = "NA")

jar <- aggregate(PropPer100k ~ Offense + Race + Year, jar, sum)

#Coerce column types
jar$Offense <- as.factor(jar$Offense)
jar$Race <- as.factor(jar$Race)
jar$Year <- as.numeric(as.character(jar$Year))


# SHINY ------------------------------------------------------------------------
# Shiny UI
ui <- shinyUI(fluidPage(
  titlePanel("Juvenile Arrest Records (1980-2015)"),
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput("race", "Select race for arrested juveniles:",
                   unique(jar$Race), selected = "American Indian"),
      radioButtons("crime", "Select crime category to examine:",
                   choices = unique(jar$Offense), selected = "All crimes")
    ),
    mainPanel(
      plotOutput("distPlot")
    )
  )
))

# Shiny server
#writing server function
server <- shinyServer(function(input, output) {
  dataset <- reactive({
    jar[which(jar$Race %in% input$race & jar$Offense == input$crime), ]
  })

  output$distPlot <- renderPlot({
    p <- ggplot(dataset()) +
      geom_line(aes(x = Year, y = PropPer100k, color=Race), size=2) +
      theme_minimal() + scale_color_tableau() + xlab(" ") +
      ylab("Number arrested per 100,000")
    print(p)
  })
})

shinyApp(ui = ui, server = server)