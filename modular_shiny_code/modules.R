################################# UI FUNCTIONS #################################
#HEADER UI
headerUI <- function(id){
  ns <- NS(id) #Create namespace function with the user-provided ID
  tags$li(a(
    href = "http://www.massmutual.com",
    img(
      src = "massmutual_logo.png",
      title = "Company Home",
      height = "50px",
      width = "200px"
    ),
    style = "padding-top: 0px; padding-bottom: 0px;"),
    class = "dropdown")
}

#DASHBOARD UI
dashboardUI <- function(id){
  ns <- NS(id) #Create namespace function with the user-provided ID
  tagList(
    column(width = 12,
           fluidRow(
             tabBox(
               width = "100%",
               tabPanel(
                 title = "Earthquake Info",
                 div(uiOutput(ns("quake_info")))
               )
             ) #End tabBox
           ), #End fluidRow
           br(),
           div(
             span(
               tags$style(HTML("table.dataTable tr.selected td, table.dataTable
                               td.selected {background-color: rgba(90, 91, 91, 0.6)
                               !important;}")),
               DT::dataTableOutput(ns("quake_table")),
               style = "height:40%;"
               )
            )
    ) #End column
  ) #End tagList
}

############################ MODULE 1: LOGIN SCREEN ############################
#UI for login module
loginUI <- function(id){
  ns <- NS(id)
  tagList(
    column(
      width = 4,
      offset = 4,
      wellPanel(
        tags$head(tags$link(rel = "stylesheet", type = "text/css",
                            href = "custom.css")),
        h4(textOutput(ns("login_info")), align = "center"),
        h3(textInput(ns("user"), label = "Username")),
        tagAppendAttributes(
          h3(passwordInput(ns("pw"), label = "Password")),
          `data-proxy-click` = ns("login")
        ),
        actionButton(inputId = ns("login"), label = "Login",
                     class = "loginButton")
      )
    )
  )
}

#Server logic for login module
login <- function(input, output, session){
  approved_users <- data.frame(users, passwords)

  login_attempt <- reactiveValues(
    message = "Please enter your login credentials", status = FALSE
  )
  output$logged_in <- reactive({
                                return(login_attempt$status)
                              })
  outputOptions(output, "logged_in", suspendWhenHidden = FALSE)

  observeEvent(
    eventExpr = input$login,
    handlerExpr = {
      validate(
        need(input$user, "Please enter your username"),
        need(input$pw, "Please enter your password")
      )
      is_approved <- approved_users %>%
        filter(users == input$user, passwords == input$pw)
      if(nrow(is_approved) > 0){
        login_attempt$status <- TRUE
        login_attempt$message <- paste("You are logged in as", input$user)
      } else {
        login_attempt$status <- FALSE
        showModal(modalDialog(
          title = "Invalid credentials",
          "Please check your login information and try again.",
          easyClose = TRUE
        ))
      }
    } #End handlerExpr
  ) #End observeEvent

  observeEvent(
    eventExpr = input$logout,
    handlerExpr = {
      login_attempt$status <- FALSE
      login_attempt$message <- paste("Logged out")
    }
  )
  output$login_info <- reactive({
    return(login_attempt$message)
  })
}


########################### MODULE 2: DATA TABLE ###############################
#Helper function to clean and format quakes data
table_output <- function(data){
  df <- quakes %>%
    rename("LATITUDE" = lat,
           "LONGITUDE" = long,
           "DEPTH" = depth,
           "MAGNITUDE" = mag,
           "REPORTING STATIONS" = stations
    )
  #Data table aesthetics
  out <- datatable(
    df, selection = "single", rownames = FALSE,
    options = list(
      pageLength = 10,
      autoWidth = FALSE,
      scrollX = FALSE,
      columnDefs = list(
        list(
          width = "200px",
          targets = c(0, 1, 2, 3, 4),
          className = "dt-center",
          targets = 0
        )
      ),
      searching = TRUE
    ),
    class = "cell-border stripe"
  )
  return(out)
}

#Server function to display quakes data on the dashboard
display_table <- function(input, output, session, data){
  output$quake_table <- DT::renderDataTable({
    table_output(data())
  })
}

############################ MODULE 3: QUAKE INFO TAB ##########################
quake_info <- function(input, output, session, data){
  output$quake_info <- renderUI({
    validate(
      need(
        try(input$quake_table_rows_selected),
        "Select a row to see information about the earthquake"
      )
    )
    num_rows <- length(input$quake_table_rows_selected)
    if(num_rows > 0){
      index <- input$quake_table_rows_selected
      selected_quake <- quakes[index, ]
      out_html <- HTML(
        "<ul style = 'list-style-type:none'>",
        "<li><b>Location: </b>", selected_quake$lat, ", ", selected_quake$long, "</li>",
        "<li><b>Depth: </b>", selected_quake$depth, "km", "</li>",
        "<li><b>Magnitude: </b>", selected_quake$mag, "(on the Richter scale)", "</li>",
        "<li><b>Number of reporting stations: </b>", selected_quake$stations, "</li>",
        "</ul>"
      )
      return(out_html)
    }
  })
}