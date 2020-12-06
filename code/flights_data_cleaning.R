install.packages('countrycode')
library(dplyr)
library(data.table)
library(janitor)

# Read & Structure the countries data
countries <- read.table('countries.dat', header=FALSE, sep = ",")
countries <- countries %>% rename("country_name"=V1, "country_iso_code"=V2, "country_dafif_code"=V3)
countries <- select(countries, -c(country_dafif_code))

# Read & Structure the airlines data
airlines <- read.table('airlines.dat', header=FALSE, sep = ",")
airlines <- airlines %>% rename("airline_id"=V1,"airline_name"=V2, "airline_alias"=V3, 
                                "airline_iata_code"=V4,"airline_icao_code"=V5, "airline_callsign"=V6,
                                "airline_country"=V7,"airline_is_operational"=V8)
airlines <- select(airlines, -c(airline_alias,airline_icao_code,airline_callsign))

# Read & Structure the aircraft data
aircraft <- read.table('aircraft.dat', header=FALSE, sep = ",")
aircraft <- aircraft %>% rename("aircraft_name"=V1,"aircraft_iata_code"=V2, "aircraft_icao_code"=V3)
aircraft <- select(aircraft, -c(aircraft_icao_code))

# Read & Structure the airports data
airports <- read.table('airports.dat', header=FALSE, sep = ",")
airports <- airports %>% rename("airport_id"=V1,"airport_name"=V2, "airport_city"=V3,
                                "airport_country"=V4,"airport_iata_code"=V5,
                                "airport_icao_code"=V6, "airport_latitude"=V7,"airport_longitude"=V8,
                                "airport_altitude"=V9,"airport_timeoffset"=V10,"aiport_daylightsaving"=V11,
                                "airport_timezone"=V12,"type"=V13,"data_source"=V14)
airports <- select(airports, 
                   -c(airport_icao_code,airport_timeoffset,aiport_daylightsaving, airport_timezone))

# Read & Structure the routes data
routes <- read.table('routes.dat', header=FALSE, sep = ",")
routes <- routes %>% rename("airline_iatacode"=V1,"airline_id"=V2, "source_airport_iata_code"=V3,
                            "source_airport_id"=V4,"dest_airport_iata_code"=V5,
                            "dest_airport_id"=V6, "route_codeshare"=V7,"route_stops"=V8,
                            "aircraft_iata_code"=V9)
# Joining 1
# + airline name
d1 <- merge(x=routes, y=airlines[,c("airline_id","airline_name")], by="airline_id")

# + source airport
d1 <- merge(x=d1,y=airports[,c("airport_iata_code","airport_name","airport_city","airport_country",
                               "airport_latitude","airport_longitude")], 
            by.x="source_airport_iata_code", by.y="airport_iata_code")
d1 <- d1 %>% rename("source_airport_name"="airport_name", "source_airport_city"="airport_city",
                    "source_airport_country"="airport_country","source_airport_latitude"="airport_latitude",
                    "source_airport_longitude"="airport_longitude")
d1 <- select(d1, -c(airline_iatacode,source_airport_iata_code,source_airport_id))

# + destination airport
d1 <- merge(x=d1,y=airports[,c("airport_iata_code","airport_name","airport_city","airport_country",
                               "airport_latitude","airport_longitude")], 
            by.x="dest_airport_iata_code", by.y="airport_iata_code")
d1 <- d1 %>% rename("dest_airport_name"="airport_name", "dest_airport_city"="airport_city",
                    "dest_airport_country"="airport_country","dest_airport_latitude"="airport_latitude",
                    "dest_airport_longitude"="airport_longitude")
d1 <- select(d1, -c(airline_id,dest_airport_iata_code,dest_airport_id))

# + aircraft
d1 <- merge(x=d1,y=aircraft, by="aircraft_iata_code")
d1 <- select(d1, -c(route_codeshare,route_stops,aircraft_iata_code))


###
d1$source_country_code <- countrycode(d1$source_airport_country, origin = 'country.name', destination = 'iso3c')
d1$dest_country_code <- countrycode(d1$dest_airport_country, origin = 'country.name', destination = 'iso3c')

# Convert to factors
d1 <- d1 %>% mutate_all(as.factor)

write.csv(d1, 'flights_clean1.csv')