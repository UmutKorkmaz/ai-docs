import { BrowserRouter as Router, Routes, Route, useParams } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './components/Home';
import MarkdownViewer from './components/MarkdownViewer';
import './App.css';

function DocViewer() {
  const { '*': docPath } = useParams();
  return <MarkdownViewer filePath={docPath} />;
}

function App() {
  return (
    <Router basename="/ai-docs">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="doc/*" element={<DocViewer />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
