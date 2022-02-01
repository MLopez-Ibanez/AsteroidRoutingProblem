library(tidyverse)
library(xtable)

xtable_sanitize_highlight <- function(x, alpha = 0.01) {
  if(!is.numeric(x)) return(sanitize(x, type = "latex"))
  return(ifelse(x < alpha, paste0("\\hls{", formatC(x,format="e", digits = 2), "}"),
                formatC(x,format="e", digits = 2)))
}
  
get.end.fitness <- function (x) {
  return ((x %>% group_by(seed) %>% summarise(.groups='drop_last', min.fitness=min(Fitness)))$min.fitness)
}

read.data <- function (folder, instance, er, alg, distance = "") {
  if (distance != "") distance <- paste0("-", distance)
  filename <- paste0(alg, distance,".csv.xz")
  if (er != "") er <- paste0("-er", er)
  subfolder <- paste0('m400', er)
  path <- paste(folder, subfolder, instance, filename, sep='/')
  return(read.csv(path))
}

read_end_fitness <- function(...) {
  return(get.end.fitness(read.data(...)))
}

compute.bb.stats <- function() {
  result <- data.frame(instance=character(), 
                       cego.bb.repr=numeric(), 
                       umm.bb.repr=numeric(), 
                       best.bb.alg=numeric())
  
  # Ranking is er=0, Order is er=1
  for (ins in instances) {
    cego_ranking <- read_end_fitness('results', ins, 0, 'cego', 'maxmindist')
    cego_order <- read_end_fitness('results', ins, 1, 'cego', 'maxmindist')
    umm_ranking <- read_end_fitness('results', ins, 0, 'umm', 'maxmindist')
    umm_order <- read_end_fitness('results', ins, 1, 'umm', 'maxmindist')
    randomsearch <- read_end_fitness('results', ins, "", 'randomsearch', '')
    
    cego.bb.repr <- wilcox.test(cego_ranking, cego_order)$p.value
    umm.bb.repr <- wilcox.test(umm_ranking, umm_order)$p.value
    best.bb.alg <- wilcox.test(cego_order, umm_ranking)$p.value

    result <- rbind(result, cbind.data.frame(instance = ins,
                                  cego.bb.repr = cego.bb.repr,
                                  umm.bb.repr = umm.bb.repr,
                                  best.bb.alg = best.bb.alg))
  }
  res_latex <- result %>% mutate(across(everything(), xtable_sanitize_highlight))
  print(xtable(res_latex, display=c("d","s","e","e","e")), include.rownames=FALSE,
        sanitize.text.function = function(x) x)
  return (result) 
}

compute.informed.stats <- function() {
  result <- data.frame(instance=character(), umm.rank.greedy=numeric(), cego.order.greedy=numeric())
  
  # Ranking is er=0, Order is er=1
  for (ins in instances) {
    cego_bb <- read_end_fitness('results', ins, 1, 'cego', 'maxmindist')
    cego_greedy <- read_end_fitness('results', ins, 1, 'cego', 'greedy_euclidean')
    umm_bb <- read_end_fitness('results', ins, 0, 'umm', 'maxmindist')
    umm_greedy <- read_end_fitness('results', ins, 0, 'umm', 'greedy_euclidean')
    
    umm.rank.greedy <- wilcox.test(umm_bb, umm_greedy)$p.value
    cego.order.greedy <- wilcox.test(cego_bb, cego_greedy)$p.value
    result <- rbind(result, cbind.data.frame(instance = ins,
                                             umm.rank.greedy = umm.rank.greedy,
                                             cego.order.greedy = cego.order.greedy))
  }
  res_latex <- result %>% mutate(across(everything(), xtable_sanitize_highlight))
  print(xtable(res_latex,display=c("d","s","e","e")), include.rownames=FALSE,
        sanitize.text.function = function(x) x)
  
  return (result) 
}

instances <- character()
for (s in c(10,15,20,25,30)) {
  for (seed in c(42,73)) {
    instances <- c(instances, paste("arp", s, seed, sep='_'))
  }
}

compute.bb.stats()
compute.informed.stats()
