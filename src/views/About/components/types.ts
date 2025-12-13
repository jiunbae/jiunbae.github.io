interface Advisor {
  name: string
  url: string
}

export interface School {
  name: string
  location: string
  degree: string
  period: string
  url: string
  status?: string
  advisor?: Advisor
  description?: string
}

export interface Education {
  schools: School[]
}

export interface Job {
  company: string
  companyUrl: string
  team?: string
  teamUrl?: string
  position: string
  period: string
  description?: string[]
}

export interface Experience {
  jobs: Job[]
}

export interface Project {
  title: string
  url?: string
  organization: string
  period: string
  description: string[]
}

export interface Projects {
  projects: Project[]
}

export interface Award {
  title: string
  year: string
}

export interface Awards {
  awards: Award[]
}

export interface Publication {
  title: string
  authors: string[]
  journal: string
  volume?: string
  year: number
  url?: string
}

export interface Publications {
  papers: Publication[]
} 