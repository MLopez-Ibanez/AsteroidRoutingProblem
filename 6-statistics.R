library(tidyverse)
read.data <- function (folder, instance, er, alg, distance) {
  subfolder <- paste('m400-er',er,sep='')
  path <- paste(folder, subfolder, instance, sep='/')
  filename <- paste(alg,distance,sep='-')
  filename <- paste(filename,'csv.xz', sep='.')
  path <- paste(path,filename,sep='/')
  data <- read.csv(path)
  return(data)
}

get.end.fitness <- function (data) {
  return ((data %>% group_by(seed) %>% summarise(.groups='drop_last', min.fitness=min(Fitness)))$min.fitness)
}

instances <- character()
for (s in c(10,15,20,25,30)) {
  for (seed in c(42,73)) {
    instances <- c(instances, paste("arp",s,seed, sep='_'))
  }
}

compute.bb.stats <- function() {
  result <- data.frame(instance=character(), cego.bb.repr=numeric(), umm.bb.repr=numeric(), 
                       ranking.bb.alg=numeric(), order.bb.alg=numeric())
  
  # Ranking is er=0, Order is er=1
  # FIXME: not sure about the above
  for (i in 1:length(instances)) {
    cego_ranking <- read.data('results', instances[i], 0, 'cego', 'maxmindist')
    cego_order <- read.data('results', instances[i], 1, 'cego', 'maxmindist')
    umm_ranking <- read.data('results', instances[i], 0, 'umm', 'maxmindist')
    umm_order <- read.data('results', instances[i], 1, 'umm', 'maxmindist')
    
    cego.bb.repr <- wilcox.test(get.end.fitness(cego_ranking), get.end.fitness(cego_order))$p.value
    umm.bb.repr <- wilcox.test(get.end.fitness(umm_ranking), get.end.fitness(umm_order))$p.value
    ranking.bb.alg <- wilcox.test(get.end.fitness(cego_ranking), get.end.fitness(umm_ranking))$p.value
    order.bb.alg <- wilcox.test(get.end.fitness(cego_order), get.end.fitness(umm_order))$p.value
    
    result[nrow(result)+1,] = c(instances[i], cego.bb.repr, umm.bb.repr, ranking.bb.alg, order.bb.alg)
    
  }
  
  return (result) 
}

compute.informed.stats <- function() {
  result <- data.frame(instance=character(), umm.order.greedy=numeric(), cego.rank.greedy=numeric())
  
  # Ranking is er=0, Order is er=1
  for (i in 1:length(instances)) {
    cego_bb <- read.data('results', instances[i], 0, 'cego', 'maxmindist')
    cego_greedy <- read.data('results', instances[i], 0, 'cego', 'greedy_euclidean')
    umm_bb <- read.data('results', instances[i], 1, 'umm', 'maxmindist')
    umm_greedy <- read.data('results', instances[i], 1, 'umm', 'greedy_euclidean')
    
    umm.order.greedy <- wilcox.test(get.end.fitness(umm_bb), get.end.fitness(umm_greedy))$p.value
    cego.rank.greedy <- wilcox.test(get.end.fitness(cego_bb), get.end.fitness(cego_greedy))$p.value
    
    result[nrow(result)+1,] = c(instances[i], umm.order.greedy, cego.rank.greedy)
    
  }
  
  return (result) 
}