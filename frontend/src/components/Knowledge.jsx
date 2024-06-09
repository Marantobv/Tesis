import React from 'react'
import { SiReact,SiHtml5,SiCss3,SiJavascript } from 'react-icons/si'
import { RiEnglishInput } from 'react-icons/ri'

function Knowledge() {
  return (
    <div className='section pt-[5rem]' id='knowledge'>
      <div className='container max-w-[1200px] mx-auto'>
      <div className=''>
            <svg width="0" height="0">
              <linearGradient id="base-gradient" x1="100%" y1="100%" x2="0%" y2="0%">
                <stop stopColor="#e042b1" offset="23%" />
                <stop stopColor="#5FC3E4" offset="53%" />
                <stop stopColor="#E55D87" offset="108%" />
              </linearGradient>
            </svg>
            <div className='grid lg:grid-cols-3 grid-cols-2'>
              <div className='lg:p-0 p-4'>
                <SiHtml5 className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}}></SiHtml5>
                <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>HTML5</p>
                <ul className='text-white text-center font-secondary text-xl'>
                  <li>SEO oriented</li>
                  <li>Following new HTML standards</li>
                </ul>
              </div>
              <div className='lg:p-0 p-4'>
                <SiCss3 className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiCss3>
                <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>CSS3</p>
                <ul className='text-white text-center font-secondary text-xl'>
                  <li>Responsive Web Design</li>
                  <li>Bootstrap</li>
                  <li>Tailwind</li>
                </ul>
              </div>
              <div className='lg:p-0 p-4'>
                <SiJavascript className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiJavascript>
                <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>JAVASCRIPT</p>
                <ul className='text-white text-center font-secondary text-xl'>
                  <li>Fetch API</li>
                  <li>DOM Manipulation</li>
                  <li>Async Programming</li>
                </ul>
              </div>
              <div className='lg:p-0 p-4'>
                <SiReact className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiReact>
                <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>REACT JS</p>
                <ul className='text-white text-center font-secondary text-xl'>
                  <li>React Hooks & Props</li>
                  <li>React Router</li>
                  <li>Material UI</li>
                </ul>
              </div>
              <div className='lg:p-0 p-4'>
                <RiEnglishInput className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></RiEnglishInput>
                <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>ENGLISH</p>
                <ul className='text-white text-center font-secondary text-xl'>
                  <li>B2 LEVEL</li>
                </ul>
              </div>
            </div>
            <h2 className='text-center font-secondary font-bold text-3xl lg:mt-[50px] mt-[20px] text-[#C637A0] uppercase'>and more coming...</h2>
          </div>
      </div>
    </div>
  )
}

export default Knowledge


{/* <div className=''>
<svg width="0" height="0">
  <linearGradient id="base-gradient" x1="100%" y1="100%" x2="0%" y2="0%">
    <stop stopColor="#e042b1" offset="23%" />
    <stop stopColor="#5FC3E4" offset="53%" />
    <stop stopColor="#E55D87" offset="108%" />
  </linearGradient>
</svg>
<div className='grid lg:grid-cols-3 grid-cols-2'>
  <div className='lg:p-0 p-4'>
    <SiHtml5 className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}}></SiHtml5>
    <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>HTML5</p>
    <ul className='text-white text-center font-secondary text-xl'>
      <li>SEO oriented</li>
      <li>Following new HTML standards</li>
    </ul>
  </div>
  <div className='lg:p-0 p-4'>
    <SiCss3 className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiCss3>
    <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>CSS3</p>
    <ul className='text-white text-center font-secondary text-xl'>
      <li>Responsive Web Design</li>
      <li>Bootstrap</li>
      <li>Tailwind</li>
    </ul>
  </div>
  <div className='lg:p-0 p-4'>
    <SiJavascript className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiJavascript>
    <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>JAVASCRIPT</p>
    <ul className='text-white text-center font-secondary text-xl'>
      <li>Fetch API</li>
      <li>DOM Manipulation</li>
      <li>Async Programming</li>
    </ul>
  </div>
  <div className='lg:p-0 p-4'>
    <SiReact className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></SiReact>
    <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>REACT JS</p>
    <ul className='text-white text-center font-secondary text-xl'>
      <li>React Hooks & Props</li>
      <li>React Router</li>
      <li>Material UI</li>
    </ul>
  </div>
  <div className='lg:p-0 p-4'>
    <RiEnglishInput className='mx-auto w-[80px] h-[80px] lg:w-[125px] lg:h-[125px]' style={{fill : 'url(#base-gradient)'}} size={145}></RiEnglishInput>
    <p className='font-primary uppercase text-center p-3 text-[#9b33cc] text-2xl'>ENGLISH</p>
    <ul className='text-white text-center font-secondary text-xl'>
      <li>B2 LEVEL</li>
    </ul>
  </div>
</div>
<h2 className='text-center font-secondary font-bold text-3xl lg:mt-[50px] mt-[20px] text-[#C637A0] uppercase'>and more coming...</h2>
</div> */}