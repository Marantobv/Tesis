import Navbar from './components/Navbar'
import Header from './components/Header'
import Home from './components/Home'
import Contact from './components/Contact'
import About from './components/About'
import Knowledge from './components/Knowledge'
import Projects from './components/Projects'
import Charts from './components/Charts'

function App() {

  return (
    <div className='bg-base'>
      <Header></Header>
      <Navbar></Navbar>
      <Home></Home>
      <About></About>
      <Charts></Charts>
      <Knowledge></Knowledge>
      <Projects></Projects>
      <Contact></Contact>
      <div className='h-[200px]'></div>
    </div>
  )
}

export default App