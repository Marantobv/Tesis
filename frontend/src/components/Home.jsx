import React from 'react'
import stock1 from './../img/stock1.png'
import { BsGithub,BsLinkedin,BsFacebook,BsWhatsapp } from 'react-icons/bs'
import { TypeAnimation } from 'react-type-animation'
import { motion } from 'framer-motion'
import { fade } from '../utils/variants'
import {Link} from 'react-scroll'

function Home() {
  return (
    <div className='h-screen flex' id='home'>
      <div className='flex flex-1 items-center justify-center lg:gap-[100px] max-w-full'>
        <div className='flex-1 flex-col lg:justify-center lg:text-left text-center p-16'>
          <motion.h2 variants={fade("up", 1)} initial={"init"} whileInView={"view"} className='lg:text-7xl text-5xl font-bold uppercase font-primary text-morado'>
          Plataforma de Análisis de Sentimientos y Predicción de Índices Bursátiles
          </motion.h2>
          <motion.div variants={fade("up", 1.3)} initial={"init"} whileInView={"view"}>
            <div  className='flex lg:justify-start justify-center items-center mt-6 mb-2 lg:mt-4 lg:mb-4 uppercase font-primary'>
              <TypeAnimation
                sequence={[
                  "Uso PLN",
                  2000,
                  "Predigo el mercado",
                  2000,
                  "Mejoro decisiones",
                  2000
                ]}
                wrapper="span"
                cursor={true}
                repeat={Infinity}
                className='lg:text-5xl text-4xl font-bold text-verde'
              />
            </div >
            <p className='lg:text-2xl text-xl mt-6 lg:my-0 font-tertiary font-light text-morado'>
            Utilizando Procesamiento de Lenguaje Natural (PLN) para predecir índices del mercado de valores y optimizar decisiones de inversión basadas en el análisis de sentimientos.
            </p>
            {/* <div className='flex lg:justify-start justify-center gap-7 lg:my-6 mt-6 mb-8'>
              <a href='https://github.com/OsmarGilCusipuma' target='_blank' aria-label='GitHub'>
                <BsGithub className='cursor-pointer onhover' size={32}></BsGithub>
              </a>
              <a href='https://www.linkedin.com/in/osmar-antony-gil-cusipuma-97097a262/' target='_blank'>
                <BsLinkedin className='cursor-pointer onhover' size={32} aria-label='Linkedin'></BsLinkedin>
              </a>
              <a href='https://www.facebook.com/antonygc2002/' target='_blank'>
                <BsFacebook className='cursor-pointer onhover' size={32} aria-label='Facebook'></BsFacebook>
              </a>
              <a href='https://wa.link/1ayc0i' target='_blank'>
                <BsWhatsapp className='cursor-pointer onhover' size={32} aria-label='Whatsapp'></BsWhatsapp>
              </a>
            </div> */}
            <div className='flex lg:justify-start justify-center items-center lg:gap-8 gap-6 mt-8'>
              <a href='https://wa.link/1ayc0i' target='_blank'>
                <button type='button' className='btn font-tertiary gradient'>Regístrate</button>
              </a>
              <a target='_blank' href='/Gil Cusipuma.pdf' className=' hover:tracking-wider transition-all cursor-pointer text-xl font-bold text-morado'>Inicia sesión</a>
            </div>
          </motion.div>
        </div>
        <div className='flex-1'>
          <motion.img variants={fade("down", 1)} initial={"init"} whileInView={"view"} className='hidden lg:flex' src={stock1} alt='Antony Dev'></motion.img>
        </div>
      </div>
    </div>
  )
}

export default Home