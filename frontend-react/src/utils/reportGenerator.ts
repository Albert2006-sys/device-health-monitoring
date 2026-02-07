import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { AnalysisResult } from '../types/analysis';

export const generatePDFReport = (result: AnalysisResult, filename: string) => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // ===== HEADER =====
    doc.setFillColor(10, 14, 39);
    doc.rect(0, 0, pageWidth, 45, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.text('Device Health Report', 15, 22);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(160, 174, 192);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 15, 32);
    doc.text(`File: ${filename}`, 15, 38);

    // ===== HEALTH SCORE =====
    const healthColor = result.health_score >= 70 ? [0, 200, 120] : [255, 0, 85];

    doc.setFontSize(48);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(healthColor[0], healthColor[1], healthColor[2]);
    doc.text(result.health_score.toString(), 40, 75, { align: 'center' });

    doc.setFontSize(12);
    doc.setTextColor(100, 100, 100);
    doc.text('HEALTH SCORE', 40, 85, { align: 'center' });

    // ===== STATUS BADGE =====
    const statusColor = result.status === 'normal' ? [0, 200, 120] : [255, 0, 85];
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
    doc.text(`Status: ${result.status.toUpperCase()}`, 130, 65, { align: 'center' });

    if (result.failure_type) {
        doc.setFontSize(11);
        doc.setTextColor(100, 100, 100);
        doc.setFont('helvetica', 'normal');
        const faultLabel = result.failure_type.replace(/_/g, ' ').toUpperCase();
        doc.text(`Fault: ${faultLabel}`, 130, 75, { align: 'center' });
    }

    // ===== CONFIDENCE =====
    const confColor = (result.confidence || 0) >= 0.85 ? [0, 200, 120] : (result.confidence || 0) >= 0.6 ? [255, 200, 0] : [255, 0, 85];
    doc.setFontSize(12);
    doc.setTextColor(confColor[0], confColor[1], confColor[2]);
    doc.setFont('helvetica', 'bold');
    doc.text(`Confidence: ${((result.confidence || 0) * 100).toFixed(0)}%`, 130, 85, { align: 'center' });

    // ===== METRICS TABLE =====
    autoTable(doc, {
        startY: 100,
        head: [['Metric', 'Value', 'Interpretation']],
        body: [
            [
                'Anomaly Score',
                result.anomaly_score.toFixed(4),
                result.anomaly_score > (result.reasoning_data?.threshold || 0.05) ? 'Above threshold' : 'Within normal range'
            ],
            [
                'Threshold',
                (result.reasoning_data?.threshold || 0.05).toFixed(4),
                'Decision boundary for anomaly detection'
            ],
            [
                'Windows Analyzed',
                (result.reasoning_data?.windows_analyzed || 0).toString(),
                '1-second segments processed'
            ],
            [
                'Anomalous Windows',
                (result.reasoning_data?.anomalous_windows || 0).toString(),
                `${(((result.reasoning_data?.anomalous_windows || 0) / (result.reasoning_data?.windows_analyzed || 1)) * 100).toFixed(0)}% of total`
            ],
            [
                'Processing Time',
                `${result.processing_ms} ms`,
                'Total inference latency'
            ],
        ],
        theme: 'grid',
        headStyles: {
            fillColor: [0, 150, 200],
            textColor: [255, 255, 255],
            fontStyle: 'bold',
        },
        styles: {
            fontSize: 10,
            cellPadding: 4,
        },
    });

    // ===== AI REASONING =====
    const finalY = (doc as any).lastAutoTable.finalY + 15;

    doc.setFontSize(14);
    doc.setTextColor(0, 0, 0);
    doc.setFont('helvetica', 'bold');
    doc.text('AI Analysis Reasoning:', 15, finalY);

    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);
    doc.setFont('helvetica', 'normal');

    const reasoning = result.explanation;
    const splitReasoning = doc.splitTextToSize(reasoning, pageWidth - 30);
    doc.text(splitReasoning, 15, finalY + 10);

    // ===== RECOMMENDATION =====
    const recommendY = finalY + 10 + (splitReasoning.length * 5) + 15;

    doc.setFillColor(0, 150, 200, 0.1);
    doc.roundedRect(15, recommendY, pageWidth - 30, 25, 3, 3, 'F');

    doc.setFontSize(11);
    doc.setTextColor(0, 0, 0);
    doc.setFont('helvetica', 'bold');
    doc.text('Recommended Action:', 20, recommendY + 8);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    const recommendation = result.status === 'normal'
        ? 'Continue routine monitoring. No immediate maintenance required.'
        : result.health_score < 50
            ? 'CRITICAL: Schedule immediate inspection and maintenance.'
            : 'Schedule maintenance within 7 days. Monitor health trend.';
    doc.text(recommendation, 20, recommendY + 18);

    // ===== FOOTER =====
    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.setFont('helvetica', 'italic');
    doc.text(
        'Device Health Monitoring System | Condition-Based Maintenance',
        pageWidth / 2,
        pageHeight - 10,
        { align: 'center' }
    );

    // ===== SAVE =====
    doc.save(`health_report_${filename.replace(/\.[^/.]+$/, '')}_${Date.now()}.pdf`);
};
