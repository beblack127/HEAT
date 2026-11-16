const React = require('react'); const RDS = require('react-dom/server'); const sharp = require('sharp');
const fa = require('react-icons/fa');
const list = { data: 'FaDatabase', mapping: 'FaMapMarkedAlt', cleaning: 'FaBroom', truth: 'FaWalking', forecast: 'FaChartLine', evaluate: 'FaClipboardCheck', heat: 'FaFire', calendar: 'FaCalendarAlt', search: 'FaSearch', subway: 'FaSubway', store: 'FaStore', brain: 'FaBrain', history: 'FaHistory', wave: 'FaWaveSquare', plus: 'FaPlusCircle', mobility: 'FaChartLine' };
(async () => { for (const [k, n] of Object.entries(list)) {
  const svg = RDS.renderToStaticMarkup(React.createElement(fa[n], { color: '#FFFFFF', size: 256 }));
  await sharp(Buffer.from(svg)).resize(256, 256).png().toFile(`${process.argv[2] || 'fig/icons'}/${k}.png`); } console.log('ok'); })();
