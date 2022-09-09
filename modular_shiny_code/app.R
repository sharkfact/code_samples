source("modules.R")
source("config.R")

ui <- dashboardPage(
  dashboardHeader(title = "Making Shiny Dashboards Shine",
                  titleWidth = 550,
                  headerUI("quakes_header")
                  ),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    tags$head(
      tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
      ),
    conditionalPanel(condition = "!output['quakes-logged_in']",
                     loginUI("quakes")),
    conditionalPanel(condition = "output['quakes-logged_in']",
                     dashboardUI("quakes")
    ) #End conditionalPanel
  ) #End dashboardBody
) #End dashboardPage

server <- function(input, output, session){
  #All modules have the same ID because all three are associated with either the
  #loginUI() function or the dashboardUI() function. Modules must share ID with
  #the UI function associated with them.
  callModule(module = login, id = "quakes")
  callModule(module = display_table, id = "quakes", data = quakes)
  callModule(module = quake_info, id = "quakes", data = quakes)
}

shinyApp(ui, server)