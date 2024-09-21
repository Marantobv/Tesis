import React from 'react'
import stock1 from './../img/stock1.png'
import prueba from './../img/prueba.png'
import { BsGithub,BsLinkedin,BsFacebook,BsWhatsapp } from 'react-icons/bs'
import { TypeAnimation } from 'react-type-animation'
import { motion } from 'framer-motion'
import { fade } from '../utils/variants'
import {Link} from 'react-scroll'

function Home() {
  return (
    <div className='h-screen flex' id='home'>
      <div className='flex lg:flex-1 items-center justify-center lg:gap-[100px] max-w-full'>
        <div className='lg:flex-1 flex-col lg:justify-center lg:text-left text-center p-16'>
          <motion.h2 variants={fade("up", 1)} initial={"init"} whileInView={"view"} className='lg:text-7xl text-5xl font-bold uppercase font-primary text-morado'>
          Modelo Predictivo de Índices Bursátiles
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
          </motion.div>
        </div>
        <div className='flex-1'>
          <motion.img variants={fade("down", 1)} initial={"init"} whileInView={"view"} className='hidden lg:flex' src={prueba} alt='Antony Dev'></motion.img>
        </div>
      </div>
    </div>
  )
}

export default Home